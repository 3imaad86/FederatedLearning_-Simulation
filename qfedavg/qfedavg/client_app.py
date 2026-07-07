"""ClientApp q-FedAvg.

Specifite : avant l'entrainement local, le client calcule F_k(w) =
loss du modele GLOBAL sur la partition LOCALE. Cette valeur est envoyee
au serveur via le metric "f_k" pour ponderer l'agregation.
"""

import time

import torch
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

    # F_k = mean loss du modele global sur la partition locale, calculee
    # avant l'entrainement (forward uniquement).
    t0 = time.perf_counter()
    model.eval()
    total_loss, total_samples = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    with torch.no_grad():
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            total_loss += float(loss_fn(model(x), y).item())
            total_samples += int(y.size(0))
    f_k = total_loss / max(total_samples, 1)
    model.train()

    # entrainement standard. Le chrono a demarre avant F_k car cette passe
    # forward est obligatoire pour q-FedAvg.
    train_loss, _ = train_fn(model, trainloader, epochs, lr, device,
                             momentum=c["momentum"])
    local_time_s = time.perf_counter() - t0

    # evaluation locale + envoi de f_k au serveur
    extra = local_eval_metrics(model, valloader, device)
    extra["f_k"] = float(f_k)

    # compression optionnelle des poids envoyes
    sd = compress_if_enabled(model.state_dict(), c)

    return make_train_reply(
        msg, sd, train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, delay, dropped=0,
        extra_metrics=extra,
        # la passe F_k est forward-only (eval) -> coute ~0.5 epoch, pas 1.0
        compute_epochs=float(epochs) + 0.5,
        model_name=model_name,
    )
