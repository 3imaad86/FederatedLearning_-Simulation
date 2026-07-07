"""ClientApp FedAvg (partitionnement IID ou non-IID configurable)."""

import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compress_if_enabled, compute_comm_delay, compute_tier_epochs,
    local_eval_metrics, make_train_reply, read_common_config,
    round_loader_seed,
)
from fl_common.data import get_device, get_model, load_data, set_seed
from fl_common.training import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    # seed differente pour chaque client, mais stable d'un round a l'autre
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)
    tier, epochs = compute_tier_epochs(
        pid, c["base_epochs"], c["epochs_hetero"], c["hardware_hetero"])
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = msg.content["arrays"].to_torch_state_dict()

    # nom du modele actif (Net ou BigNet)
    model_name = str(context.run_config.get("model-name", "net")).lower().strip()

    # profil reseau et delai de communication
    net_tier, delay = compute_comm_delay(
        pid, c["downlink_comm_size_ratio"], c["sim_model_mb"], seed=c["seed"],
        model_name=model_name,
        uplink_comm_size_ratio=c["uplink_comm_size_ratio"])

    # entrainement local
    model = get_model(model_name)
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, valloader = load_data(pid, c["num_parts"], c["bs"],
                                       data_hetero=c["data_hetero"],
                                       partitioning=c["partitioning"],
                                       alpha=c["dir_alpha"],
                                       seed=c["data_seed"],
                                       loader_seed=round_loader_seed(c, round_idx))
    t0 = time.perf_counter()
    train_loss, _ = train_fn(model, trainloader, epochs, lr, device,
                             momentum=c["momentum"])
    local_time_s = time.perf_counter() - t0

    # evaluation locale sur la partition du client
    extra = local_eval_metrics(model, valloader, device)

    # compression optionnelle des poids envoyes
    sd = compress_if_enabled(model.state_dict(), c)
    return make_train_reply(
        msg, sd, train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, delay, dropped=0,
        extra_metrics=extra,
        model_name=model_name,
    )
