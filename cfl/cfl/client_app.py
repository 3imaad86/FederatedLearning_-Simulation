"""ClientApp CFL (Customized Federated Learning, Wang et al. 2023).

Tous les clients voient les memes images CIFAR-10. CFL adapte uniquement
la taille du submodel au tier hardware du client.

Le client :
  1. Lit son `width_ratio` (decide par le serveur via Algo 1).
  2. Charge sa partition CIFAR-10 standard.
  3. Construit un SubNet aux dimensions reduites et y charge les params
     du parent (prefix-keep des K premiers canaux par layer).
  4. Entraine localement E epochs en SGD standard.
  5. Renvoie son sub-state_dict + (w_k, acc_k) pour que le serveur :
        - agrege via Algo 3 (zero-pad + masked average)
        - entraine le predictor via Algo 2.
"""

import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compress_if_enabled,
    compute_comm_delay, compute_tier_epochs,
    local_eval_metrics, make_train_reply, read_common_config,
    round_loader_seed,
)
from fl_common.data import get_device, load_data, set_seed
from fl_common.training import train as train_fn

from .submodel import (
    build_submodel,
    parent_to_submodel_state_dict,
)

app = ClientApp()


def _resolve_width(msg: Message, pid: int) -> float:
    """Lit le width assigne au pid dans `cfl-widths-per-pid`.

    Fallback : si le serveur n'a pas envoye la cle (ne devrait pas arriver
    avec CFLStrategy), on prend 1.0 (full model) pour rester compatible.
    """
    widths = msg.content["config"].get("cfl-widths-per-pid")
    if widths is None:
        return 1.0
    try:
        return float(widths[int(pid)])
    except (IndexError, TypeError, ValueError):
        return 1.0


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

    # nom du modele parent (Net ou BigNet)
    model_name = str(context.run_config.get("model-name", "net")).lower().strip()

    # width decide par le serveur pour ce client
    width_ratio = _resolve_width(msg, pid)

    parent_sd = msg.content["arrays"].to_torch_state_dict()

    # profil reseau et delai de communication
    net_tier, delay = compute_comm_delay(
        pid, c["downlink_comm_size_ratio"], c["sim_model_mb"], seed=c["seed"],
        model_name=model_name,
        uplink_comm_size_ratio=c["uplink_comm_size_ratio"],
        payload_scale=float(width_ratio) ** 2)

    # construit le SubNet et charge les params du parent (prefix-keep)
    model = build_submodel(width_ratio, model_name)
    sub_sd = parent_to_submodel_state_dict(parent_sd, width_ratio, model_name)
    model.load_state_dict(sub_sd, strict=True)
    device = get_device()
    model.to(device)

    trainloader, valloader = load_data(
        pid, c["num_parts"], c["bs"],
        data_hetero=c["data_hetero"],
        partitioning=c["partitioning"],
        alpha=c["dir_alpha"],
        seed=c["data_seed"],
        loader_seed=round_loader_seed(c, round_idx),
    )

    # entrainement SGD local standard
    t0 = time.perf_counter()
    train_loss, _ = train_fn(
        model, trainloader, epochs, lr, device,
        momentum=c["momentum"],
    )
    local_time_s = time.perf_counter() - t0

    # evaluation locale + metriques CFL pour le serveur
    extra = local_eval_metrics(model, valloader, device)
    extra["cfl_width_ratio"] = float(width_ratio)

    # FLOPs Conv2d ~ width^2 : l'energie compute facturee suit ce ratio.
    compute_scale = float(width_ratio) ** 2

    # compression optionnelle des poids envoyes
    sd = compress_if_enabled(model.state_dict(), c)
    return make_train_reply(
        msg,
        sd,  # sub-state_dict (channels reduits) + compression eventuelle
        train_loss,
        len(trainloader.dataset),
        local_time_s,
        pid, tier, epochs, net_tier, delay,
        dropped=0, extra_metrics=extra,
        model_name=model_name,
        compute_scale=compute_scale,
    )
