"""Helpers partages par les ClientApp.

Deux profils de simulation reseau :
  A) Algos straggler-aware (HFL-BEFL) : decide_early_drop + finalize_comm
     (drops reseau, deadline, sleep).
  B) Algos straggler-free (FedAvg, FedProx, ...) : compute_comm_delay,
     juste un (net_tier, delay) pour la mesure d'energie radio.
"""

import math
import time

import torch
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from .compression import apply_compression as _apply_compression
from .compression import effective_size_ratio as _eff_size_ratio
from .data import model_size_bytes
from .energy import compute_energy_components, configure_energy_model
from .straggler import (
    mbps_transfer_time_s,
    network_profile,
    roundtrip_delay_s,
    simulate_comm_delay,
)
from .training import test as _test


def compress_if_enabled(state_dict, c, skip_prefixes=None):
    """Applique la compression definie par read_common_config (no-op si desactivee)."""
    bits = int(c.get("compression_bits", 32))
    sparsity = float(c.get("compression_sparsity", 0.0))
    if bits >= 32 and sparsity <= 0.0:
        return state_dict
    return _apply_compression(state_dict, bits=bits, sparsity=sparsity,
                              skip_prefixes=skip_prefixes)


def effective_comm_size_ratio(c):
    """Ratio uplink effectif = ratio manuel * ratio de compression."""
    manual = float(c.get("comm_size_ratio_manual",
                         c.get("downlink_comm_size_ratio",
                               c.get("comm_size_ratio", 1.0))))
    bits = int(c.get("compression_bits", 32))
    sparsity = float(c.get("compression_sparsity", 0.0))
    return manual * _eff_size_ratio(bits, sparsity)


def read_common_config(context):
    """Params communs a tous les client_app, lus depuis context."""
    cfg = context.run_config
    configure_energy_model(cfg)
    num_parts = int(context.node_config["num-partitions"])
    cfg_num_clients = int(cfg.get("num-clients", num_parts))
    if cfg_num_clients != num_parts:
        raise ValueError(
            "Configuration incoherente: num-clients="
            f"{cfg_num_clients} mais Flower expose num-partitions={num_parts}. "
            "Aligne `num-clients` avec `options.num-supernodes` dans la "
            "configuration Flower pour eviter des metriques trompeuses."
        )
    comm_ratio_manual = float(cfg.get("comm-size-ratio", 1.0))
    compression_ratio = _eff_size_ratio(
        int(cfg.get("compression-quantization-bits", 32)),
        float(cfg.get("compression-sparsity-ratio", 0.0)),
    )
    uplink_ratio = comm_ratio_manual * compression_ratio
    return {
        "pid": context.node_config["partition-id"],
        "num_parts": num_parts,
        "bs": cfg["batch-size"],
        "base_epochs": int(cfg.get("local-epochs", 1)),
        "data_hetero": int(cfg.get("data-heterogeneity", 0)),
        "epochs_hetero": int(cfg.get("epochs-heterogeneity", 0)),
        "hardware_hetero": int(
            cfg.get("hardware-heterogeneity",
                    int(cfg.get("epochs-heterogeneity", 0)))
        ),
        "partitioning": str(cfg.get("partitioning", "noniid")),
        "dir_alpha": float(cfg.get("dirichlet-alpha", 0.3)),
        "straggler_sim": int(cfg.get("straggler-sim", 0)),
        "round_deadline": float(cfg.get("round-deadline-s", 0.0)),
        "fedstrag_enabled": int(cfg.get("fedstrag-enabled", 0)),
        # Ratios downlink/uplink separes : la compression quant/sparse ne
        # reduit que l'uplink, le broadcast serveur reste au ratio manuel.
        "comm_size_ratio_manual": comm_ratio_manual,
        "compression_ratio": compression_ratio,
        "comm_size_ratio": uplink_ratio,
        "downlink_comm_size_ratio": comm_ratio_manual,
        "uplink_comm_size_ratio": uplink_ratio,
        # Taille modele simulee en MB (0 = vraie taille).
        "sim_model_mb": float(cfg.get("sim-model-mb", 0.0)),
        # momentum SGD : 0.0 par defaut (FedNova et SCAFFOLD exigent 0).
        "momentum": float(cfg.get("momentum", 0.0)),
        # Seed runtime (-1 = pas de seeding).
        "seed": int(cfg.get("seed", -1)),
        # Seed du partitionnement, separe du seed runtime.
        "data_seed": int(cfg.get("data-seed", 42)),
        "compression_bits": int(cfg.get("compression-quantization-bits", 32)),
        "compression_sparsity": float(cfg.get("compression-sparsity-ratio", 0.0)),
    }


def round_loader_seed(config, round_idx):
    """Seed DataLoader reproductible mais different a chaque round."""
    seed = int(config.get("seed", -1))
    if seed < 0:
        return None
    return seed + int(round_idx) * 10007


def compute_tier_epochs(pid, base_epochs, epochs_hetero, hardware_hetero=None):
    """Tier compute (0/1/2) + nb d'epochs (heterogeneite hardware).

    hardware_hetero=1 avec epochs_hetero=0 expose des tiers weak/medium/strong
    sans changer le nombre d'epochs (utile pour CFL).
    """
    if hardware_hetero is None:
        hardware_hetero = epochs_hetero
    tier = int(pid) % 3 if int(hardware_hetero) else 1
    if not epochs_hetero:
        return tier, int(base_epochs)
    epochs = {0: 1, 1: 3, 2: 6}[tier]   # ratio 6x (stable)
    return tier, epochs


def _effective_model_mb(comm_size_ratio, sim_model_mb, model_name="net"):
    """Taille effective transmise apres compression (en MB)."""
    base = float(sim_model_mb) if sim_model_mb > 0 else (
        model_size_bytes(model_name) / (1024.0 * 1024.0))
    return base * float(comm_size_ratio)


def decide_early_drop(straggler_sim, pid, round_idx, comm_size_ratio=1.0,
                      sim_model_mb=0.0, seed=-1):
    """Retourne (net_tier, delay, dropped_early).

    `delay` est toujours calcule (pour l'energie) mais le sleep n'a lieu
    que si straggler-sim=1 (via finalize_comm). dropped_early=True = panne
    reseau, skipper l'entrainement.
    """
    model_mb = _effective_model_mb(comm_size_ratio, sim_model_mb)
    if straggler_sim:
        net_tier, delay = simulate_comm_delay(
            pid, model_mb, round_idx, seed=seed)
        if delay is None:
            return net_tier, None, True
        return net_tier, delay, False
    # Mode rapide : pas de drop, pas de sleep, juste le delay pour l'energie.
    net_tier, bw, rtt, _, _ = network_profile(pid, seed=seed)
    return net_tier, roundtrip_delay_s(model_mb, bw, rtt), False


def compute_comm_delay(pid, comm_size_ratio=1.0, sim_model_mb=0.0, seed=-1,
                       model_name="net", uplink_comm_size_ratio=None,
                       payload_scale=1.0, downlink_payload_scale=None,
                       uplink_payload_scale=None, extra_downlink_mb=0.0,
                       extra_uplink_mb=0.0):
    """Retourne (net_tier, delay) sans simulation de stragglers.

    Pour les algos sans drops/deadline (FedAvg, FedProx, ...). Le delay sert
    uniquement a calculer l'energie radio, aucun sleep n'est applique.

    `payload_scale` (et ses variantes down/up) reduit les bytes transferes
    (ex: CFL width^2, SCAFFOLD 2x). `extra_*_mb` sont des MB additionnels
    deja effectifs (ex: logits FedAvg+KL).
    """
    base_scale = float(payload_scale)
    down_scale = (
        base_scale if downlink_payload_scale is None
        else float(downlink_payload_scale)
    )
    up_scale = (
        base_scale if uplink_payload_scale is None
        else float(uplink_payload_scale)
    )
    down_mb = (
        _effective_model_mb(comm_size_ratio, sim_model_mb, model_name)
        * down_scale
        + max(0.0, float(extra_downlink_mb))
    )
    up_ratio = (comm_size_ratio if uplink_comm_size_ratio is None
                else uplink_comm_size_ratio)
    up_mb = (
        _effective_model_mb(up_ratio, sim_model_mb, model_name)
        * up_scale
        + max(0.0, float(extra_uplink_mb))
    )
    net_tier, bw, rtt, _, _ = network_profile(pid, seed=seed)
    delay = (
        mbps_transfer_time_s(down_mb, bw)
        + mbps_transfer_time_s(up_mb, bw)
        + float(rtt)
    )
    return net_tier, delay


def _deadline_downlink_time(delay, downlink_time_s=None):
    """Downlink deja consomme quand l'upload rate la deadline (~ moitie du round-trip)."""
    if downlink_time_s is not None:
        return max(0.0, float(downlink_time_s))
    return max(0.0, float(delay)) * 0.5


def finalize_comm(delay, local_time_s, round_deadline, downlink_time_s=None):
    """Simule la communication, avec short-circuit si la deadline depasse.

    Retourne (comm_time_s, dropped_deadline).
    """
    if round_deadline > 0 and (local_time_s + delay) > round_deadline:
        return _deadline_downlink_time(delay, downlink_time_s), True
    time.sleep(delay)
    return delay, False


def _staleness_info(delay, local_time_s, round_deadline):
    """Metriques de retard pour FedStrag (staleness en nombre de deadlines depassees)."""
    total_s = max(0.0, float(local_time_s)) + max(0.0, float(delay))
    deadline = float(round_deadline)
    if deadline <= 0.0 or total_s <= deadline:
        return {
            "fedstrag_late": 0.0,
            "fedstrag_staleness": 0.0,
            "deadline_miss_s": 0.0,
            "server_wait_time_s": float(total_s),
        }
    return {
        "fedstrag_late": 1.0,
        "fedstrag_staleness": float(max(1, int(math.ceil(total_s / deadline)) - 1)),
        "deadline_miss_s": float(total_s - deadline),
        # Le serveur attend au plus la deadline ; l'update arrive plus tard.
        "server_wait_time_s": float(deadline),
    }


def finalize_comm_fedstrag(delay, local_time_s, round_deadline,
                           accept_late=False, downlink_time_s=None):
    """Finalize communication avec option FedStrag.

    Retourne (comm_time_s, dropped_deadline, extra_metrics).
    accept_late=True conserve l'update en retard (annotee avec sa staleness)
    au lieu de la dropper.
    """
    extra = _staleness_info(delay, local_time_s, round_deadline)
    if extra["fedstrag_late"] >= 0.5:
        if accept_late:
            return float(delay), False, extra
        return _deadline_downlink_time(delay, downlink_time_s), True, extra
    time.sleep(delay)
    return float(delay), False, extra


def local_eval_metrics(net, loader, device):
    """Eval du modele entraine sur la partition locale du client.

    En non-IID ce score surestime l'accuracy reelle ; compare a l'eval
    serveur, il visualise le biais non-IID.
    """
    loss, acc = _test(net, loader, device)
    return {"local_eval_loss": float(loss), "local_eval_acc": float(acc)}


# Cache global du loader testset (1 instance par process Ray).
_GLOBAL_TEST_LOADER = None


def global_test_eval_metrics(net, device, batch_size=256, sample_size=0,
                             exclude_indices=None):
    """Eval du modele sur le testset global CIFAR-10.

    sample_size > 0 limite l'eval aux N premieres images (deterministe).
    exclude_indices retire des images du testset (ex: proxy FedAvg+KL).
    """
    global _GLOBAL_TEST_LOADER
    cache_key = (int(sample_size),
                 tuple(sorted(int(i) for i in (exclude_indices or []))))
    if (_GLOBAL_TEST_LOADER is None
            or getattr(_GLOBAL_TEST_LOADER, "_cache_key", None) != cache_key):
        from torch.utils.data import DataLoader, Subset
        from .data import get_testset, get_testset_excluding_indices
        if exclude_indices:
            ds = get_testset_excluding_indices(exclude_indices)
        else:
            ds = get_testset()
        if sample_size > 0 and sample_size < len(ds):
            ds = Subset(ds, list(range(int(sample_size))))
        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
        loader._cache_key = cache_key
        _GLOBAL_TEST_LOADER = loader
    loader = _GLOBAL_TEST_LOADER
    loss, acc = _test(net, loader, device)
    return {"global_eval_loss": float(loss), "global_eval_acc": float(acc)}


def _build_reply(msg, state_dict, metrics):
    """Construit un Message reply standard (poids + metric record)."""
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


def _base_metrics(pid, tier, net_tier):
    """Champs metric communs aux replies drop et train."""
    return {
        "partition_id": float(pid),
        "resource_tier": float(tier),
        "net_tier": float(net_tier),
        "local_eval_loss": 0.0,
        "local_eval_acc": 0.0,
        # Flower exige que tous les MetricRecord d'un round portent les
        # memes cles, donc on initialise partout les champs conditionnels.
        "befl_drop": 0.0,
        "fedstrag_late": 0.0,
        "fedstrag_staleness": 0.0,
        "deadline_miss_s": 0.0,
        "server_wait_time_s": 0.0,
    }


def make_drop_reply(msg, global_sd, pid, tier, net_tier, extra_metrics=None,
                    link_type="wan", comm_time_s=0.0,
                    no_model_transfer=1.0):
    """Reply rapide (sans entrainement) : poids globaux inchanges, dropped=1.

    On renvoie un state_dict avec les memes cles que les replies train,
    sinon Flower rejette le round (`All ArrayRecords must have the same keys`).
    """
    compute_j, comm_j = compute_energy_components(
        tier, net_tier, 0.0, comm_time_s, link_type=link_type,
        num_examples=0, epochs=0,
    )
    energy_j = compute_j + comm_j
    m = _base_metrics(pid, tier, net_tier) | {
        "train_loss": 0.0, "num-examples": 0,
        "local_time_s": 0.0, "comm_time_s": float(comm_time_s),
        "epochs_used": 0.0, "dropped": 1.0,
        "compute_epochs_used": 0.0,
        "energy_j": float(energy_j),
        "compute_energy_j": float(compute_j),
        "comm_energy_j": float(comm_j),
        # 1.0 = aucun modele transfere (drop reseau precoce).
        # 0.0 = downlink deja eu lieu mais pas d'upload (deadline, skip BEFL).
        "no_model_transfer": float(no_model_transfer),
        "drop_deadline": 0.0,
        "link_type_lan": 1.0 if str(link_type).lower() == "lan" else 0.0,
    }
    m["server_wait_time_s"] = float(comm_time_s)
    if extra_metrics:
        m.update(extra_metrics)
    return _build_reply(msg, global_sd, m)


def make_train_reply(msg, state_dict, train_loss, num_examples, local_time_s,
                     pid, tier, epochs, net_tier, comm_time_s, dropped,
                     extra_metrics=None, link_type="wan", compute_epochs=None,
                     compute_scale=1.0, model_name=None):
    """Reply d'entrainement normal, avec energie ventilee compute/comm.

    `compute_scale` module l'energie compute selon la taille reelle du
    modele entraine (CFL : FLOPs ∝ width^2). `model_name` ajoute un facteur
    proportionnel a la taille du modele vs Net. `link_type` choisit la table
    de puissance radio ("wan" pour FedAvg/..., "lan" pour HFL device-edge).
    """
    energy_epochs = float(epochs if compute_epochs is None else compute_epochs)
    compute_j, comm_j = compute_energy_components(
        tier, net_tier, local_time_s, comm_time_s, link_type=link_type,
        num_examples=num_examples, epochs=energy_epochs,
        compute_scale=compute_scale,
        model_name=model_name)
    energy_j = compute_j + comm_j
    m = _base_metrics(pid, tier, net_tier) | {
        "train_loss": float(train_loss),
        "num-examples": int(num_examples),
        "local_time_s": float(local_time_s),
        "comm_time_s": float(comm_time_s),
        "epochs_used": float(epochs),
        "compute_epochs_used": float(energy_epochs),
        "dropped": float(dropped),
        "energy_j": float(energy_j),
        "compute_energy_j": float(compute_j),
        "comm_energy_j": float(comm_j),
        "no_model_transfer": 0.0,
        "drop_deadline": float(dropped),
        # 1.0 = lan, 0.0 = wan (MetricRecord ne supporte que int/float).
        "link_type_lan": 1.0 if str(link_type).lower() == "lan" else 0.0,
    }
    m["server_wait_time_s"] = float(local_time_s) + float(comm_time_s)
    if extra_metrics:
        m.update(extra_metrics)
    return _build_reply(msg, state_dict, m)
