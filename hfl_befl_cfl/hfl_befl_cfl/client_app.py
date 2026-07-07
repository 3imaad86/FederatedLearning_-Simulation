"""ClientApp HFL-BEFL + CFL.

Workflow par round :
  1. Lit la config du serveur :
     - rl-edge-map[pid]         : edge assignment (RL)
     - befl-tau-per-pid[pid]    : nombre d'epochs autorise (BEFL Lyapunov)
     - cfl-widths-per-pid[pid]  : ratio width du submodel (CFL)
  2. Si tau=0 (BEFL drop) -> reply drop immediat.
  3. Sinon : construit le submodel BigNet@width et y charge les params parent.
  4. Entraine pour `min(tau, base_epochs)` epochs.
  5. Renvoie le sub-state_dict + width + acc locale pour Algo 2 CFL serveur.

NOTE : la simulation straggler LAN reste active (heritee de HFL-BEFL),
mais FedStrag est deconseille avec CFL (warning au demarrage).
"""

import random
import time

from ._local_imports import ensure_workspace_packages

ensure_workspace_packages()

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from cfl.submodel import build_submodel, parent_to_submodel_state_dict

from fl_common.client_helpers import (
    compress_if_enabled,
    compute_tier_epochs, finalize_comm_fedstrag, local_eval_metrics,
    make_train_reply, read_common_config, round_loader_seed,
)
from fl_common.compression import effective_size_ratio
from fl_common.data import get_device, load_data, model_size_bytes, set_seed
from fl_common.energy import battery_for_tier
from fl_common.straggler import mbps_transfer_time_s
from fl_common.training import train as train_fn

# Re-utilise les helpers existants de HFL-BEFL pour profil LAN, drop LAN, etc.
from hfl_befl.client_app import (
    CLIENT_EDGE_PROFILES,
    LAN_PDROP,
    battery_metrics,
    client_edge_profile,
    downlink_only_time,
    performance_edge_id,
    read_tau_from_server,
    selected_edge_id,
)
from fl_common.strategy import HFL_EDGE_PREFIX


app = ClientApp()


def tier_compression_config(common_config, run_config, tier):
    """Retourne la config compression specifique au tier hardware.

    Cles supportees dans pyproject.toml:
      tier0-compression-quantization-bits
      tier1-compression-quantization-bits
      tier2-compression-quantization-bits

    Si une cle est absente, on retombe sur la compression globale existante.

    La sparsification (top-k magnitude masking) a ete retiree car elle
    degradait significativement les performances HFL-BEFL-CFL : sparsity
    est forcee a 0.0 quoi qu'il arrive. Seule la quantification per-tier
    reste active.
    """
    t = int(tier)
    bits_key = f"tier{t}-compression-quantization-bits"
    bits = int(run_config.get(bits_key, common_config["compression_bits"]))
    if bits not in (4, 8, 16, 32):
        raise ValueError(
            f"{bits_key} doit etre 4, 8, 16 ou 32 (recu {bits})"
        )
    sparsity = 0.0  # sparsification desactivee
    compression_ratio = effective_size_ratio(bits, sparsity)
    uplink_ratio = common_config["comm_size_ratio_manual"] * compression_ratio
    out = dict(common_config)
    out["compression_bits"] = bits
    out["compression_sparsity"] = sparsity
    out["compression_ratio"] = compression_ratio
    out["comm_size_ratio"] = uplink_ratio
    out["uplink_comm_size_ratio"] = uplink_ratio
    return out


def communication_delay(pid, round_idx, straggler_sim, comm_size_ratio,
                        sim_model_mb, bandwidth, rtt, jitter_s,
                        net_tier=None, seed=-1, model_name="net",
                        width_ratio=1.0, uplink_comm_size_ratio=None):
    """Delai device -> edge avec scaling CFL (width^2 sur le payload).

    Le client transmet ~ w^2 * params du parent (Conv2d FLOPs/params en w^2).
    """
    base_mb = float(sim_model_mb)
    if base_mb <= 0:
        base_mb = model_size_bytes(model_name) / (1024.0 * 1024.0)
    # CFL : un client a width w transmet ~w^2 fois moins de bytes.
    base_mb = base_mb * float(width_ratio) ** 2
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


def edge_state_dict_from_message(msg, edge_id):
    """Extrait le modele edge depuis l'ArrayRecord packe (cf. HFL standard)."""
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


def _resolve_cfl_width(msg, pid):
    """Lit cfl-widths-per-pid[pid]. Fallback 1.0 (full)."""
    widths = msg.content["config"].get("cfl-widths-per-pid")
    if widths is None:
        return 1.0
    try:
        return float(widths[int(pid)])
    except (IndexError, TypeError, ValueError):
        return 1.0


def make_cfl_drop_reply(msg, sub_sd, pid, tier, net_tier, comm_time_s,
                        extra_metrics, no_model_transfer):
    """Drop reply avec les memes cles de sous-modele que les replies train.

    Flower valide la coherence des cles ArrayRecord avant que la strategie
    puisse ignorer les replies `dropped=1`. Un dummy `__drop_dummy__` melange
    avec des sous-modeles CFL provoque donc `InconsistentMessageReplies`.
    """
    extra = dict(extra_metrics or {})
    extra["no_model_transfer"] = float(no_model_transfer)
    return make_train_reply(
        msg,
        sub_sd,
        train_loss=0.0,
        num_examples=0,
        local_time_s=0.0,
        pid=pid,
        tier=tier,
        epochs=0,
        net_tier=net_tier,
        comm_time_s=comm_time_s,
        dropped=1,
        extra_metrics=extra,
        link_type="lan",
        compute_scale=0.0,
    )


@app.train()
def train(msg: Message, context: Context):
    # config commune + decisions serveur
    c = read_common_config(context)
    pid = c["pid"]
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)

    cfg = context.run_config
    num_edges = int(cfg.get("hfl-num-edges", 3))
    base_battery_j = float(cfg.get("befl-battery-j", 0.0))
    unlimited_tier = int(cfg.get("befl-unlimited-tier", 2))
    model_name = str(cfg.get("model-name", "bignet")).lower().strip()

    # tier compute + epochs naturels
    base_tier, base_epochs = compute_tier_epochs(
        pid, c["base_epochs"], c["epochs_hetero"], c["hardware_hetero"])
    tc = tier_compression_config(c, cfg, base_tier)
    fallback_edge = performance_edge_id(base_tier, max(1, num_edges))
    edge_id = selected_edge_id(msg, pid, fallback_edge, max(1, num_edges))

    # batterie info (state cote serveur via BEFLEdgeState)
    constrained = (int(base_tier) != int(unlimited_tier))
    battery_j = battery_for_tier(base_battery_j, base_tier) if constrained else 0.0

    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))

    # decisions serveur : tau (BEFL) + width (CFL)
    tau_from_server = read_tau_from_server(msg, pid)
    width_ratio = _resolve_cfl_width(msg, pid)
    parent_sd = edge_state_dict_from_message(msg, edge_id)
    initial_sub_sd = parent_to_submodel_state_dict(
        parent_sd, width_ratio, model_name)

    extra = battery_metrics(battery_j, constrained)
    extra["hfl_edge_id"] = float(edge_id)
    extra["cfl_width_ratio"] = float(width_ratio)
    extra["compression_bits"] = float(tc["compression_bits"])
    extra["compression_sparsity"] = float(tc["compression_sparsity"])
    extra["compression_ratio"] = float(tc["compression_ratio"])
    extra["uplink_size_ratio"] = float(tc["uplink_comm_size_ratio"])

    # BEFL drop -> retour immediat
    if tau_from_server == 0:
        net_tier_lan, edge_bw, edge_rtt, edge_jitter = client_edge_profile(pid)
        downlink_delay, dropped_lan = communication_delay(
            pid, round_idx, c["straggler_sim"], c["downlink_comm_size_ratio"],
            c["sim_model_mb"], edge_bw, edge_rtt, edge_jitter,
            net_tier=net_tier_lan, seed=c["seed"],
            model_name=model_name, width_ratio=width_ratio,
            uplink_comm_size_ratio=c["downlink_comm_size_ratio"])
        if dropped_lan:
            return make_cfl_drop_reply(
                msg, initial_sub_sd, pid, base_tier, net_tier_lan,
                comm_time_s=0.0, extra_metrics=extra,
                no_model_transfer=1.0)
        return make_cfl_drop_reply(
            msg, initial_sub_sd, pid, base_tier, net_tier_lan,
            comm_time_s=downlink_only_time(
                c["downlink_comm_size_ratio"], c["sim_model_mb"],
                edge_bw, edge_rtt, model_name=model_name,
                width_ratio=width_ratio),
            extra_metrics={**extra, "befl_drop": 1.0},
            no_model_transfer=0.0)

    # tau > 0 (decision BEFL) ou tau = -1 (1er round / unlimited) -> base_epochs.
    if tau_from_server > 0:
        epochs = min(tau_from_server, base_epochs)
    else:
        epochs = base_epochs

    # communication LAN
    net_tier, edge_bw, edge_rtt, edge_jitter = client_edge_profile(pid)
    delay, dropped_lan = communication_delay(
        pid, round_idx, c["straggler_sim"], c["downlink_comm_size_ratio"],
        c["sim_model_mb"], edge_bw, edge_rtt, edge_jitter,
        net_tier=net_tier, seed=c["seed"],
        model_name=model_name, width_ratio=width_ratio,
        uplink_comm_size_ratio=tc["uplink_comm_size_ratio"])

    if dropped_lan:
        return make_cfl_drop_reply(
            msg, initial_sub_sd, pid, base_tier, net_tier,
            comm_time_s=0.0, extra_metrics=extra,
            no_model_transfer=1.0)

    # construit le SubNet a ce width et charge les params du parent
    model = build_submodel(width_ratio, model_name)
    sub_sd = initial_sub_sd
    model.load_state_dict(sub_sd, strict=True)
    device = get_device()
    model.to(device)

    # chargement donnees + entrainement local
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

    # upload simule + deadline
    comm_time_s, dropped = delay, 0
    fedstrag_extra = {}
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline, fedstrag_extra = finalize_comm_fedstrag(
            delay, local_time_s, c["round_deadline"],
            accept_late=c["fedstrag_enabled"],
            downlink_time_s=downlink_only_time(
                c["downlink_comm_size_ratio"], c["sim_model_mb"],
                edge_bw, edge_rtt, model_name=model_name,
                width_ratio=width_ratio))
        if (not c["fedstrag_enabled"]) and dropped_deadline:
            dropped = 1
            # drop : on revient au sub_sd parent (pas d'update)
            model.load_state_dict(sub_sd)

    # evaluation locale + reply
    extra2 = local_eval_metrics(model, valloader, device)
    extra.update(extra2)
    extra.update(fedstrag_extra)

    # CFL : compute_scale = w^2 (FLOPs Conv2d scalent en w^2)
    compute_scale = float(width_ratio) ** 2

    # compression optionnelle, composable avec la reduction de width CFL
    sd = compress_if_enabled(model.state_dict(), tc)
    return make_train_reply(
        msg, sd,  # sub-state_dict (channels reduits) + compression eventuelle
        train_loss, len(trainloader.dataset),
        local_time_s, pid, base_tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
        link_type="lan",
        compute_scale=compute_scale,
        model_name=model_name,
    )
