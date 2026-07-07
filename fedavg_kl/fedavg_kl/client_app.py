"""ClientApp FedAvg+KL : KMeans local + entrainement + filtrage logits proxy.

Etapes par round :
  1. Decode (w_global, agg_logits_prev) du message et charge w_global.
  2. (round 1) Entraine KMeans local et persiste centroides + seuil T^ID.
  3. Entrainement local : CE sur les donnees privees + distillation KL sur
     le proxy si des logits agreges existent.
  4. Calcule les logits sur les indices proxy du round, filtre via
     KMeans-DRE, et renvoie (w_local, logits_filtres, masque_ID).
"""

import time

import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, Context, Message
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from fl_common.client_helpers import (
    compute_comm_delay, compute_tier_epochs,
    local_eval_metrics, make_train_reply, read_common_config,
    round_loader_seed,
)
from fl_common.data import (
    get_device, get_model, get_proxy_subset, load_data, set_seed,
)
from fl_common.training import train as train_fn

from fedavg_kl.kmeans_dre import (
    compute_id_threshold,
    filter_proxy_id_mask,
    fit_kmeans_per_class,
)
from fedavg_kl.strategy import (
    INDEX_PREFIX_CLI,
    LOGITS_PREFIX_AGG,
    LOGITS_PREFIX_CLI,
    LOGITS_VALID_PREFIX_AGG,
    MASK_PREFIX_CLI,
)

app = ClientApp()


# Cles context.state pour persistance entre rounds (KMeans)
KEY_CENTROIDS = "fedavgkl_centroids"
KEY_THRESHOLD = "fedavgkl_id_threshold"


def _split_message_arrays(combined_sd):
    """Décode (w_global, agg_logits, agg_valid_mask) du message serveur."""
    w_sd = {}
    agg_logits = None
    agg_valid_mask = None
    for k, v in combined_sd.items():
        if k == LOGITS_PREFIX_AGG:
            agg_logits = v
        elif k == LOGITS_VALID_PREFIX_AGG:
            agg_valid_mask = v
        else:
            w_sd[k] = v
    return w_sd, agg_logits, agg_valid_mask


def _fit_kmeans_if_needed(context, trainloader, num_classes, clusters_per_class,
                          id_percentile, device, seed):
    """Calcule centroides + T^ID au 1er round, persiste dans context.state.

    Returns (centroids, threshold). Les centroides sont sur `device`.
    """
    state_arrays = context.state.array_records
    state_config = context.state.config_records.setdefault(
        "fedavgkl_state", _empty_config_record())
    if (KEY_CENTROIDS in state_arrays
            and KEY_THRESHOLD in state_config):
        cents = state_arrays[KEY_CENTROIDS].to_torch_state_dict()["t"]
        thr = float(state_config[KEY_THRESHOLD])
        return cents.to(device), thr

    cents = fit_kmeans_per_class(
        trainloader,
        num_classes=num_classes,
        clusters_per_class=clusters_per_class,
        device=device,
        seed=int(seed) if int(seed) >= 0 else 42,
    )
    thr = compute_id_threshold(
        cents, trainloader, percentile=id_percentile, device=device)

    # Persiste pour les rounds suivants. ArrayRecord exige un state_dict ->
    # on encode le centroides sous la cle "t".
    context.state.array_records[KEY_CENTROIDS] = ArrayRecord({"t": cents.cpu()})
    state_config[KEY_THRESHOLD] = float(thr)
    return cents, thr


def _empty_config_record():
    """Cree un ConfigRecord vide compatible Flower."""
    from flwr.app import ConfigRecord
    return ConfigRecord({})


@torch.no_grad()
def _compute_logits_on_proxy(model, proxy_loader, device):
    """Logits du modele courant sur tout le proxy_loader ((N_proxy, K))."""
    was_training = model.training
    model.eval()
    try:
        out = []
        for x, _ in proxy_loader:
            x = x.to(device)
            logits = model(x).detach().cpu()
            out.append(logits)
        return torch.cat(out, dim=0)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def _compute_logits_and_mask_on_proxy(model, proxy_loader, centroids,
                                      id_threshold, device):
    """Calcule logits + masque ID/OOD en une seule passe sur le proxy."""
    was_training = model.training
    model.eval()
    try:
        logits_chunks = []
        mask_chunks = []
        for x, _ in proxy_loader:
            x_dev = x.to(device)
            logits = model(x_dev).detach().cpu()
            mask = filter_proxy_id_mask(
                x_dev, centroids, id_threshold).cpu()
            logits_chunks.append(logits)
            mask_chunks.append(mask)
        logits_local = torch.cat(logits_chunks, dim=0)
        id_mask_local = torch.cat(mask_chunks, dim=0).float()
        return logits_local, id_mask_local
    finally:
        if was_training:
            model.train()


def _distill_loss(student_logits, teacher_logits, T):
    """KL(student/T || teacher/T) * T^2 (Hinton et al. 2015)."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def _distill_loss_masked(student_logits, teacher_logits, T, valid_mask):
    """KL calculee uniquement sur les samples avec valid_mask[i] == 1
    (les positions sans signal teacher sont ignorees)."""
    valid_mask = valid_mask.view(-1).to(student_logits.device)
    if float(valid_mask.sum().item()) <= 0.0:
        return torch.zeros((), device=student_logits.device)
    mask_bool = valid_mask > 0.5
    s = F.log_softmax(student_logits[mask_bool] / T, dim=-1)
    t = F.softmax(teacher_logits[mask_bool] / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def _train_with_distillation(
    model, trainloader, proxy_loader, agg_logits_for_proxy,
    epochs, lr, device, distill_lambda, distill_T, momentum,
    valid_mask=None,
):
    """Entrainement local CE + KL distillation (retombe sur CE pur si
    distill_lambda <= 0 ou pas de logits teacher).

    Retourne (train_loss_moyen, num_steps).
    """
    use_distill = (
        distill_lambda > 0.0
        and agg_logits_for_proxy is not None
        and float(agg_logits_for_proxy.abs().sum().item()) > 0.0
    )
    if not use_distill:
        # Fallback : entrainement standard CE.
        return train_fn(
            model, trainloader, epochs, lr, device, momentum=momentum)

    model.to(device)
    crit_ce = torch.nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()

    agg_logits_for_proxy = agg_logits_for_proxy.to(device)
    proxy_size = int(agg_logits_for_proxy.size(0))
    if valid_mask is not None:
        valid_mask = valid_mask.to(device).float().view(-1)
    tot_loss, tot_ex, steps = 0.0, 0, 0
    proxy_iter = iter(proxy_loader)
    proxy_index_counter = 0

    for _ in range(int(epochs)):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce_loss = crit_ce(model(x), y)

            # Recupere un batch proxy + teacher logits correspondants.
            try:
                px, _ = next(proxy_iter)
            except StopIteration:
                proxy_iter = iter(proxy_loader)
                proxy_index_counter = 0
                px, _ = next(proxy_iter)
            batch_size_px = px.size(0)
            batch_start = proxy_index_counter
            batch_end = batch_start + batch_size_px
            teacher = agg_logits_for_proxy[batch_start:batch_end]
            proxy_index_counter = batch_end % proxy_size

            def _ce_only_step():
                ce_loss.backward()
                opt.step()

            if teacher.size(0) != batch_size_px:
                # Proxy mal aligne -> CE pur
                _ce_only_step()
            else:
                # determine si ce batch a du signal teacher
                if valid_mask is not None:
                    vm_slice = valid_mask[batch_start:batch_end]
                    has_signal = vm_slice.size(0) == batch_size_px \
                        and float(vm_slice.sum().item()) > 0.0
                else:
                    has_signal = True

                if not has_signal:
                    # Batch entierement vierge -> CE pur
                    _ce_only_step()
                else:
                    px = px.to(device)
                    student = model(px)
                    if valid_mask is not None:
                        kl = _distill_loss_masked(
                            student, teacher, distill_T, vm_slice)
                    else:
                        kl = _distill_loss(student, teacher, distill_T)
                    total = (1.0 - distill_lambda) * ce_loss + distill_lambda * kl
                    total.backward()
                    opt.step()

            bs = y.size(0)
            tot_loss += ce_loss.item() * bs  # log CE pure pour comparabilite
            tot_ex += bs
            steps += 1

    return tot_loss / max(tot_ex, 1), steps


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)
    tier, epochs = compute_tier_epochs(
        pid, c["base_epochs"], c["epochs_hetero"], c["hardware_hetero"])
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))

    # Hyperparams FedAvg+KL du config
    cfg_msg = msg.content["config"]
    distill_lambda = float(cfg_msg.get("fedavgkl-distill-lambda", 0.5))
    distill_T = float(cfg_msg.get("fedavgkl-distill-temperature", 4.0))
    proxy_size = int(cfg_msg.get("fedavgkl-proxy-size", 2000))
    proxy_indices_this_round = list(
        cfg_msg.get("fedavgkl-proxy-indices", list(range(proxy_size))))

    # Hyperparams KMeans-DRE fixes : 1 centroide par classe et seuil ID au
    # 90e percentile.
    clusters_per_class = 1
    id_percentile = 90.0
    proxy_seed = int(context.run_config.get("data-seed", 42))

    model_name = str(context.run_config.get("model-name", "net")).lower().strip()

    # decode (w_global, agg_logits, valid_mask) du message
    combined_sd = msg.content["arrays"].to_torch_state_dict()
    w_sd, agg_logits, agg_valid_mask = _split_message_arrays(combined_sd)

    # Profil reseau : poids du modele + surcout des logits (downlink et uplink).
    n_proxy_round = len(proxy_indices_this_round) if proxy_indices_this_round else proxy_size
    agg_logits_mb = 0.0
    if agg_logits is not None:
        agg_logits_mb = (
            agg_logits.numel() * agg_logits.element_size()
            / (1024.0 * 1024.0)
            * c["downlink_comm_size_ratio"]
        )
    logits_uplink_mb = (
        n_proxy_round
        * ((10 + 1) * 4 + 8)  # logits fp32 + mask fp32 + index int64
        / (1024.0 * 1024.0)
        * c["uplink_comm_size_ratio"]
    )
    net_tier, delay = compute_comm_delay(
        pid, c["downlink_comm_size_ratio"], c["sim_model_mb"], seed=c["seed"],
        model_name=model_name,
        uplink_comm_size_ratio=c["uplink_comm_size_ratio"],
        extra_downlink_mb=agg_logits_mb,
        extra_uplink_mb=logits_uplink_mb)

    # charge les poids globaux recus du serveur
    device = get_device()
    model = get_model(model_name)
    model.load_state_dict(w_sd)
    model.to(device)

    trainloader, valloader = load_data(
        pid, c["num_parts"], c["bs"],
        data_hetero=c["data_hetero"],
        partitioning=c["partitioning"],
        alpha=c["dir_alpha"],
        seed=c["data_seed"],
        loader_seed=round_loader_seed(c, round_idx))

    # calcule/recupere centroides KMeans + T^ID (round 1 seulement)
    centroids, id_threshold = _fit_kmeans_if_needed(
        context, trainloader,
        num_classes=10,
        clusters_per_class=clusters_per_class,
        id_percentile=id_percentile,
        device=device,
        seed=c["data_seed"] + pid,
    )

    # setup proxy loader (les MEMES indices/ordre pour tous les clients)
    proxy_ds_full, proxy_idx_list = get_proxy_subset(
        proxy_size=proxy_size, seed=proxy_seed)
    proxy_loader_full = DataLoader(
        proxy_ds_full, batch_size=c["bs"], shuffle=False)

    # Entrainement CE + distillation sur le proxy complet (les agg_logits
    # du round precedent couvrent tout le proxy).
    t0 = time.perf_counter()
    train_loss, _ = _train_with_distillation(
        model, trainloader, proxy_loader_full, agg_logits,
        epochs=epochs, lr=lr, device=device,
        distill_lambda=distill_lambda, distill_T=distill_T,
        momentum=c["momentum"],
        valid_mask=agg_valid_mask,
    )
    local_time_s = time.perf_counter() - t0

    # logits + masque ID/OOD sur le sous-proxy de ce round (en une passe)
    if proxy_indices_this_round:
        from torch.utils.data import Subset
        ordered_local_indices = sorted(
            int(i) for i in set(proxy_indices_this_round))
        sub_proxy = Subset(proxy_ds_full, ordered_local_indices)
        sub_loader = DataLoader(sub_proxy, batch_size=c["bs"], shuffle=False)
    else:
        sub_loader = proxy_loader_full
        ordered_local_indices = list(range(len(proxy_ds_full)))

    logits_local, id_mask_local = _compute_logits_and_mask_on_proxy(
        model, sub_loader, centroids, id_threshold, device)

    # Logits mis a 0 aux positions OOD ; le masque permet au serveur de
    # moyenner sans biais.
    logits_filtered = logits_local.float()
    logits_filtered = logits_filtered * id_mask_local.unsqueeze(1)

    # Reply compacte : logits/masque du sous-proxy + indices correspondants
    # (le serveur scatterise vers le proxy complet).
    proxy_indices_tensor = torch.tensor(
        ordered_local_indices, dtype=torch.long)

    # evaluation locale + diagnostics KMeans-DRE
    extra = local_eval_metrics(model, valloader, device)
    extra["fedavgkl_id_ratio"] = float(id_mask_local.mean().item()) if int(
        id_mask_local.numel()) > 0 else 0.0
    extra["fedavgkl_id_threshold"] = float(id_threshold)
    extra["fedavgkl_n_centroids"] = int(centroids.size(0))

    # pack le state dict de la reply : poids + logits + mask + indices
    reply_sd = {name: p.detach().cpu()
                for name, p in model.named_parameters()
                if p.is_floating_point()}
    # ajoute les buffers float eventuels (ex: BatchNorm)
    for name, b in model.named_buffers():
        if b.is_floating_point():
            reply_sd[name] = b.detach().cpu()
    reply_sd[LOGITS_PREFIX_CLI] = logits_filtered.cpu()
    reply_sd[MASK_PREFIX_CLI] = id_mask_local.cpu()
    reply_sd[INDEX_PREFIX_CLI] = proxy_indices_tensor

    return make_train_reply(
        msg, reply_sd, train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, delay, dropped=0,
        extra_metrics=extra,
        model_name=model_name,
    )
