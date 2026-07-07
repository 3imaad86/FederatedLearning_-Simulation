"""ClientApp HFL-BEFL.

Le client est un simple worker : il lit son tau dans `befl-tau-per-pid[pid]`
(-1 = base_epochs, 0 = drop decide par BEFL, >0 = nb d'epochs) et son edge
dans `rl-edge-map[pid]`, entraine, puis renvoie ses metriques (energie,
batterie, tier...). Toutes les decisions sont cote serveur.
"""

import random
import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compress_if_enabled,
    compute_tier_epochs, finalize_comm_fedstrag, local_eval_metrics,
    make_drop_reply, make_train_reply, read_common_config, round_loader_seed,
)
from fl_common.data import get_device, get_model, load_data, model_size_bytes, set_seed
from fl_common.energy import battery_for_tier
from fl_common.strategy import HFL_EDGE_PREFIX
from fl_common.straggler import mbps_transfer_time_s
from fl_common.training import train as train_fn

app = ClientApp()


# Profils reseau client <-> edge : net_tier -> (bandwidth_mbps, rtt_s, jitter_s).
# Liens LAN locaux, bien plus rapides que les liens WAN de FedAvg.
CLIENT_EDGE_PROFILES = {
    0: (20.0,  0.05, 0.10),    # WiFi local lent / vieux AP
    1: (100.0, 0.02, 0.05),    # WiFi local moderne
    2: (500.0, 0.01, 0.02),    # Ethernet gigabit / WiFi 6 local
}

# Probabilites de drop reseau LAN (~5x plus fiables que le WAN equivalent).
LAN_PDROP = {
    0: 0.02,   # WiFi local lent
    1: 0.005,  # WiFi local moderne
    2: 0.001,  # Ethernet
}


def performance_edge_id(tier, num_edges):
    """Cluster simple par performance: weak/medium/strong vers des edges."""
    if int(num_edges) <= 1:
        return 0
    return int(tier) % int(num_edges)


def selected_edge_id(msg, pid, fallback_edge, num_edges):
    """Lit l'edge_id depuis `rl-edge-map` envoye par le serveur (RL)."""
    edge_map = msg.content["config"].get("rl-edge-map")
    if edge_map is None:
        return int(fallback_edge)
    try:
        return int(edge_map[int(pid)]) % int(num_edges)
    except (IndexError, TypeError, ValueError):
        return int(fallback_edge)


def client_edge_profile(pid):
    """Renvoie (net_tier, bw_mbps, rtt_s, jitter_s) pour le pid."""
    net_tier = int(pid) % len(CLIENT_EDGE_PROFILES)
    bandwidth, rtt, jitter = CLIENT_EDGE_PROFILES[net_tier]
    return net_tier, bandwidth, rtt, jitter


def communication_delay(pid, round_idx, straggler_sim, comm_size_ratio,
                        sim_model_mb, bandwidth, rtt, jitter_s,
                        net_tier=None, seed=-1, model_name="net",
                        uplink_comm_size_ratio=None):
    """Delai device -> edge (round-trip). Retourne (delay_s, dropped_lan).

    dropped_lan (panne LAN) n'est actif que sous straggler_sim=1.
    """
    base_mb = float(sim_model_mb)
    if base_mb <= 0:
        base_mb = model_size_bytes(model_name) / (1024.0 * 1024.0)
    down_mb = base_mb * float(comm_size_ratio)
    up_ratio = (comm_size_ratio if uplink_comm_size_ratio is None
                else uplink_comm_size_ratio)
    up_mb = base_mb * float(up_ratio)
    if float(bandwidth) <= 0.0:
        raise ValueError("bandwidth doit etre > 0")
    delay = (
        mbps_transfer_time_s(down_mb, bandwidth)
        + mbps_transfer_time_s(up_mb, bandwidth)
        + float(rtt)
    )
    dropped_lan = False
    if int(straggler_sim):
        seed_base = int(seed) if seed is not None and int(seed) >= 0 else 2024
        rng = random.Random(seed_base + int(pid) * 1009 + int(round_idx) * 9176)
        delay += rng.uniform(0.0, max(0.0, float(jitter_s)))
        if net_tier is not None:
            pdrop = LAN_PDROP.get(int(net_tier), 0.0)
            if pdrop > 0 and rng.random() < pdrop:
                dropped_lan = True
    return delay, dropped_lan


def downlink_only_time(comm_size_ratio, sim_model_mb, bandwidth, rtt,
                       model_name="net", width_ratio=1.0):
    """Temps du downlink deja paye: bytes downlink + demi-RTT."""
    base_mb = float(sim_model_mb)
    if base_mb <= 0:
        base_mb = model_size_bytes(model_name) / (1024.0 * 1024.0)
    down_mb = base_mb * float(width_ratio) ** 2 * float(comm_size_ratio)
    return mbps_transfer_time_s(down_mb, bandwidth) + float(rtt) * 0.5


def read_tau_from_server(msg, pid):
    """Lit tau dans `befl-tau-per-pid[pid]`. Retourne -1 si pas envoye."""
    taus = msg.content["config"].get("befl-tau-per-pid")
    if taus is None:
        return -1
    try:
        value = taus[int(pid)]
        if value is None:
            return -1
        return int(value)
    except (IndexError, TypeError, ValueError):
        return -1


def edge_state_dict_from_message(msg, edge_id):
    """Extrait le modele de l'edge depuis l'ArrayRecord packe
    (cles `__hfl_edge_<id>__...`). Message non packe -> state_dict tel quel."""
    full_sd = msg.content["arrays"].to_torch_state_dict()
    marker = f"{HFL_EDGE_PREFIX}{int(edge_id)}__"
    if not any(k.startswith(HFL_EDGE_PREFIX) for k in full_sd):
        return full_sd
    selected = {
        k[len(marker):]: v
        for k, v in full_sd.items()
        if k.startswith(marker)
    }
    if not selected:
        raise ValueError(f"modele edge_id={edge_id} absent du message HFL")
    return selected


def battery_metrics(battery_j, constrained):
    """Le client renvoie juste sa capacite ; le serveur tracke energy_used."""
    return {
        "battery_constrained": float(constrained),
        "battery_capacity_j": float(battery_j),
        "battery_remaining_j": -1.0,  # tracke cote serveur, ignore ici
        "battery_remaining": 1.0,
    }


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)

    cfg = context.run_config
    num_edges = int(cfg.get("hfl-num-edges", 3))
    base_battery_j = float(cfg.get("befl-battery-j", 0.0))
    unlimited_tier = int(cfg.get("befl-unlimited-tier", 2))
    # nom du modele actif (Net ou BigNet)
    model_name = str(cfg.get("model-name", "net")).lower().strip()

    # tier compute + edge assignment (decide cote serveur)
    base_tier, base_epochs = compute_tier_epochs(
        pid, c["base_epochs"], c["epochs_hetero"], c["hardware_hetero"])
    fallback_edge = performance_edge_id(base_tier, max(1, num_edges))
    edge_id = selected_edge_id(msg, pid, fallback_edge, max(1, num_edges))

    # batterie (capacite), envoyee au serveur en metric
    constrained = (int(base_tier) != int(unlimited_tier))
    battery_j = battery_for_tier(base_battery_j, base_tier) if constrained else 0.0

    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = edge_state_dict_from_message(msg, edge_id)

    # decision BEFL recue du serveur
    tau_from_server = read_tau_from_server(msg, pid)

    extra = battery_metrics(battery_j, constrained)
    extra["hfl_edge_id"] = float(edge_id)

    # tau=0 -> drop decide par BEFL, sans entrainement. Le downlink est
    # compte (le modele a deja ete envoye) et befl_drop=1 indique au RL de
    # ne pas penaliser la policy pour cette decision.
    if tau_from_server == 0:
        net_tier_lan, edge_bw, edge_rtt, edge_jitter = client_edge_profile(pid)
        downlink_delay, dropped_lan = communication_delay(
            pid, round_idx, c["straggler_sim"], c["downlink_comm_size_ratio"],
            c["sim_model_mb"], edge_bw, edge_rtt, edge_jitter,
            net_tier=net_tier_lan, seed=c["seed"], model_name=model_name,
            uplink_comm_size_ratio=c["downlink_comm_size_ratio"])
        if dropped_lan:
            return make_drop_reply(
                msg, global_sd, pid, base_tier, net_tier_lan,
                extra_metrics=extra,
                link_type="lan")
        return make_drop_reply(
            msg, global_sd, pid, base_tier, net_tier_lan,
            extra_metrics={**extra, "befl_drop": 1.0},
            link_type="lan",
            comm_time_s=downlink_only_time(
                c["downlink_comm_size_ratio"], c["sim_model_mb"],
                edge_bw, edge_rtt, model_name=model_name),
            no_model_transfer=0.0)

    # tau > 0 = nombre d'epochs decide par BEFL (clamp au base_epochs
    # naturel), tau = -1 = pas de decision -> base_epochs.
    if tau_from_server > 0:
        epochs = min(tau_from_server, base_epochs)
    else:
        epochs = base_epochs

    # communication simulee + entrainement local
    net_tier, edge_bw, edge_rtt, edge_jitter = client_edge_profile(pid)
    delay, dropped_lan = communication_delay(
        pid, round_idx, c["straggler_sim"], c["downlink_comm_size_ratio"],
        c["sim_model_mb"], edge_bw, edge_rtt, edge_jitter,
        net_tier=net_tier, seed=c["seed"], model_name=model_name,
        uplink_comm_size_ratio=c["uplink_comm_size_ratio"])
    # Drop reseau LAN avant entrainement : le modele n'a pas ete recu.
    if dropped_lan:
        return make_drop_reply(
            msg, global_sd, pid, base_tier, net_tier, extra_metrics=extra,
            link_type="lan")

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

    # upload + check deadline
    comm_time_s, dropped = delay, 0
    fedstrag_extra = {}
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline, fedstrag_extra = finalize_comm_fedstrag(
            delay, local_time_s, c["round_deadline"],
            accept_late=c["fedstrag_enabled"],
            downlink_time_s=downlink_only_time(
                c["downlink_comm_size_ratio"], c["sim_model_mb"],
                edge_bw, edge_rtt, model_name=model_name))
        # Avec FedStrag, un retard deadline devient une update late
        # bufferisee cote serveur, pas un drop client.
        if (not c["fedstrag_enabled"]) and dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)

    # reply au serveur (energy_j est calcule par make_train_reply)
    extra = local_eval_metrics(model, valloader, device)
    extra.update(fedstrag_extra)
    extra.update(battery_metrics(battery_j, constrained))
    extra["hfl_edge_id"] = float(edge_id)

    # compression optionnelle des poids envoyes
    sd = compress_if_enabled(model.state_dict(), c)
    return make_train_reply(
        msg, sd, train_loss, len(trainloader.dataset),
        local_time_s, pid, base_tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
        model_name=model_name,
        # Lien LAN local (WiFi/Ethernet) vers un edge -> table de puissance LAN.
        link_type="lan",
    )
