"""ClientApp SCAFFOLD : training avec correction par control variates.

Le client recoit (w_global, c_global) packes dans le meme ArrayRecord avec
prefix __cg__, et maintient son c_local persistant via context.state (cle
versionnee par le partitionnement courant pour eviter les bugs cross-config).

Etapes par round :
  1. Decode (w_global, c_global) du message
  2. Recupere c_local du context.state (zeros au 1er round)
  3. Train SCAFFOLD : y = y - lr * (grad + c_global - c_local)
  4. Calcule c_local_new = c_local - c_global + (1/(K*lr)) * (w_global - y)
  5. Calcule delta_c = c_local_new - c_local
  6. Sauve c_local_new dans context.state
  7. Renvoie (y, delta_c) packes (prefix __dc__)
"""

import time

import torch
from flwr.app import ArrayRecord, Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compute_comm_delay, compute_tier_epochs,
    local_eval_metrics, make_train_reply, read_common_config,
    round_loader_seed,
)
from fl_common.data import get_device, get_model, load_data, set_seed
from fl_common.strategy import CG_PREFIX, DC_PREFIX
from fl_common.training import train_scaffold

app = ClientApp()


def _split_arrays(combined_sd):
    """Decode (w_global, c_global) depuis l'ArrayRecord combine."""
    w_sd, c_global_sd = {}, {}
    for k, v in combined_sd.items():
        if k.startswith(CG_PREFIX):
            c_global_sd[k[len(CG_PREFIX):]] = v
        else:
            w_sd[k] = v
    return w_sd, c_global_sd


def _c_local_key(c, model_name="net", lr=None):
    """Cle de stockage versionnee par configuration : si le partitionnement
    change entre deux runs, l'ancien c_local n'a plus de sens."""
    lr_suffix = f"_lr{float(lr):.8g}" if lr is not None else ""
    return (
        f"c_local_n{c['num_parts']}"
        f"_p{c['partitioning']}"
        f"_a{c['dir_alpha']:.4f}"
        f"_dh{c['data_hetero']}"
        f"_eh{c['epochs_hetero']}"
        f"_hh{c.get('hardware_hetero', c['epochs_hetero'])}"
        f"_e{c['base_epochs']}"
        f"_s{c['seed']}"
        f"_ds{c.get('data_seed', 42)}"
        f"_m{str(model_name).lower().strip()}"
        f"{lr_suffix}"
    )


def _get_or_init_c_local(context, c_global_sd, key):
    """Lit c_local depuis context.state. Si absent ou incompatible, init a zeros."""
    if key in context.state.array_records:
        cached = context.state.array_records[key].to_torch_state_dict()
        same_keys = set(cached.keys()) == set(c_global_sd.keys())
        same_shapes = same_keys and all(
            cached[k].shape == c_global_sd[k].shape for k in c_global_sd
        )
        if same_keys and same_shapes:
            return cached
        print(
            f"[scaffold] WARN: c_local persiste pour key={key!r} incompatible "
            f"avec c_global (keys_match={same_keys}, shapes_match={same_shapes}). "
            "Reinit a zeros."
        )
    return {name: torch.zeros_like(t) for name, t in c_global_sd.items()}


def _save_c_local(context, c_local_sd, key):
    """Sauve c_local pour le round suivant."""
    context.state.array_records[key] = ArrayRecord(c_local_sd)


def _pack_y_and_delta_c(model, delta_c_sd):
    """Pack y (parametres flottants, pas les buffers) + delta_c (prefix __dc__)."""
    combined = {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.is_floating_point()
    }
    for name, t in delta_c_sd.items():
        combined[f"{DC_PREFIX}{name}"] = t
    return combined


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

    # SCAFFOLD exige vanilla SGD (la correction diverge avec momentum).
    if c["momentum"] != 0.0:
        raise ValueError(
            f"SCAFFOLD exige momentum=0.0 (theorie Karimireddy 2020 basee sur "
            f"vanilla SGD). Recu momentum={c['momentum']} cote client. "
            "Aligne `momentum` dans `pyproject.toml` du run."
        )

    # nom du modele actif (Net ou BigNet)
    model_name = str(context.run_config.get("model-name", "net")).lower().strip()

    # decode (w_global, c_global) du message
    combined_sd = msg.content["arrays"].to_torch_state_dict()
    w_global_sd, c_global_sd = _split_arrays(combined_sd)

    # profil reseau et delai de communication
    net_tier, delay = compute_comm_delay(
        pid, c["downlink_comm_size_ratio"], c["sim_model_mb"], seed=c["seed"],
        model_name=model_name,
        uplink_comm_size_ratio=c["comm_size_ratio_manual"],
        downlink_payload_scale=2.0,
        uplink_payload_scale=2.0)

    # setup model + recupere c_local persistant (cle versionnee par config)
    model = get_model(model_name)
    model.load_state_dict(w_global_sd)
    device = get_device()
    model.to(device)
    state_key = _c_local_key(c, model_name=model_name, lr=lr)
    c_local_sd = _get_or_init_c_local(context, c_global_sd, state_key)

    # sauve w_global pour calculer (w - y) apres training
    w_global_copy = {name: t.detach().clone() for name, t in w_global_sd.items()}

    # train SCAFFOLD avec correction
    trainloader, valloader = load_data(pid, c["num_parts"], c["bs"],
                                       data_hetero=c["data_hetero"],
                                       partitioning=c["partitioning"],
                                       alpha=c["dir_alpha"],
                                       seed=c["data_seed"],
                                       loader_seed=round_loader_seed(c, round_idx))
    t0 = time.perf_counter()
    train_loss, num_steps = train_scaffold(
        model, trainloader, epochs, lr, device, c_global_sd, c_local_sd,
        momentum=c["momentum"],
    )
    local_time_s = time.perf_counter() - t0

    # c_new = c_old - c_global + (1/(K*lr)) * (w_global - y).
    # Si num_steps == 0 (loader vide), on garde c_local tel quel.
    y_sd = {name: p.detach().cpu() for name, p in model.named_parameters()
            if p.is_floating_point()}
    if num_steps <= 0:
        new_c_local_sd = {name: t.clone() for name, t in c_local_sd.items()}
    else:
        coef = 1.0 / (num_steps * lr)
        new_c_local_sd = {}
        for name in c_local_sd:
            if name in y_sd:
                new_c_local_sd[name] = (
                    c_local_sd[name] - c_global_sd[name]
                    + coef * (w_global_copy[name] - y_sd[name])
                )
            else:
                new_c_local_sd[name] = c_local_sd[name]

    # delta_c = c_new - c_old (pour aggregation serveur)
    delta_c_sd = {name: new_c_local_sd[name] - c_local_sd[name]
                  for name in c_local_sd}

    # sauve c_local pour le prochain round
    _save_c_local(context, new_c_local_sd, state_key)

    # evaluation locale sur la partition du client
    extra = local_eval_metrics(model, valloader, device)

    # pack y + delta_c dans un seul state_dict (prefix __dc__)
    full_state = _pack_y_and_delta_c(model, delta_c_sd)

    return make_train_reply(
        msg, full_state, train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, delay, dropped=0,
        extra_metrics=extra,
        model_name=model_name,
    )
