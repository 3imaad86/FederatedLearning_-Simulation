"""Boucle serveur partagee par tous les algos FL du repo.

Pour chaque round :
  1. strategy.start(num_rounds=1) -> train cote clients
  2. server_evaluate()             -> eval sur le test set CIFAR-10
  3. log_round() + print_round()
  4. early stopping (optionnel)

L'eval se fait cote serveur car en non-IID les metriques clients
surestiment l'accuracy (chaque client overfitte sa partition).
"""

import logging
import os
import shutil
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from torch.utils.data import DataLoader

from .data import (
    get_device, get_model, get_proxy_subset, get_testset,
    get_testset_excluding_indices, model_size_bytes, set_seed,
)
from .energy import configure_energy_model
from .metrics import (
    ensure_dir, get_results_dir, log_client_details, log_participation,
    log_round, log_summary, reset_files, resolve_dst_results_dir,
    rounds_to_convergence, rounds_to_target, set_results_dir,
)
from .training import test_with_class_accuracies


def parse_config(cfg):
    """Lit run_config et calcule la taille modele effective (compression)."""
    sim_mb = float(cfg.get("sim-model-mb", 0.0))
    manual_ratio = float(cfg.get("comm-size-ratio", 1.0))
    # Compression optionnelle : le ratio uplink effectif = manual * compression.
    from .compression import effective_size_ratio as _eff
    quant_bits = int(cfg.get("compression-quantization-bits", 32))
    sparsity = float(cfg.get("compression-sparsity-ratio", 0.0))
    comp_ratio = _eff(quant_bits, sparsity)
    downlink_ratio = manual_ratio
    uplink_ratio = manual_ratio * comp_ratio
    model_name = str(cfg.get("model-name", "net")).lower().strip()
    base_mb = sim_mb if sim_mb > 0 else model_size_bytes(model_name) / (1024.0 * 1024.0)
    parsed = {
        "compression_bits": quant_bits,
        "compression_sparsity": sparsity,
        "compression_ratio": comp_ratio,
        "comm_size_ratio_manual": manual_ratio,
        "model_name": model_name,
        "num_rounds": int(cfg["num-server-rounds"]),
        "lr": float(cfg["learning-rate"]),
        "num_clients": int(cfg.get("num-clients", 10)),
        "partitioning": str(cfg.get("partitioning", "noniid")),
        "dir_alpha": float(cfg.get("dirichlet-alpha", 0.3)),
        "hardware_hetero": int(
            cfg.get("hardware-heterogeneity",
                    int(cfg.get("epochs-heterogeneity", 0)))
        ),
        "es_patience": int(cfg.get("early-stopping-patience", 0)),
        "es_min_delta": float(cfg.get("early-stopping-min-delta", 0.001)),
        "straggler_sim": int(cfg.get("straggler-sim", 0)),
        "round_deadline_s": float(cfg.get("round-deadline-s", 0.0)),
        "fedstrag_enabled": int(cfg.get("fedstrag-enabled", 0)),
        "fedstrag_alpha": float(cfg.get("fedstrag-alpha", 1.0)),
        "fedstrag_min_weight": float(cfg.get("fedstrag-min-weight", 0.05)),
        "fedstrag_max_staleness": int(cfg.get("fedstrag-max-staleness", 0)),
        "comm_size_ratio": uplink_ratio,
        "downlink_comm_size_ratio": downlink_ratio,
        "uplink_comm_size_ratio": uplink_ratio,
        "sim_model_mb": sim_mb,
        "model_mb": base_mb * uplink_ratio,
        "downlink_model_mb": base_mb * downlink_ratio,
        "uplink_model_mb": base_mb * uplink_ratio,
        "seed": int(cfg.get("seed", -1)),
        "data_seed": int(cfg.get("data-seed", 42)),
        "momentum": float(cfg.get("momentum", 0.0)),
        # FedAvg+KL : exclure les images proxy du test set pour eviter une
        # fuite train/eval via distillation.
        "fedavgkl_proxy_size": int(cfg.get("fedavgkl-proxy-size", 0)),
        "fedavgkl_eval_exclude_proxy": int(cfg.get("fedavgkl-eval-exclude-proxy", 1)),
        # Fraction de clients echantillonnes par round (< 1.0 = cross-device).
        "fraction_train": float(cfg.get("fraction-train", 1.0)),
    }
    validate_common_config(parsed, cfg)
    return parsed


def validate_common_config(parsed, raw_cfg):
    """Valide les invariants communs avant de lancer Flower/Ray."""
    if parsed["num_rounds"] <= 0:
        raise ValueError("num-server-rounds doit etre > 0")
    if parsed["lr"] <= 0.0:
        raise ValueError("learning-rate doit etre > 0")
    if parsed["num_clients"] <= 0:
        raise ValueError("num-clients doit etre > 0")
    if parsed["partitioning"].lower() not in {"iid", "noniid", "noniid-balanced"}:
        raise ValueError(
            "partitioning doit etre 'iid', 'noniid' ou 'noniid-balanced'"
        )
    if int(raw_cfg.get("batch-size", 1)) <= 0:
        raise ValueError("batch-size doit etre > 0")
    if int(raw_cfg.get("local-epochs", 1)) <= 0:
        raise ValueError("local-epochs doit etre > 0")
    if parsed["partitioning"].lower() != "iid" and parsed["dir_alpha"] <= 0.0:
        raise ValueError("dirichlet-alpha doit etre > 0 hors partitioning=iid")
    if parsed["es_patience"] < 0:
        raise ValueError("early-stopping-patience doit etre >= 0")
    if parsed["es_min_delta"] < 0.0:
        raise ValueError("early-stopping-min-delta doit etre >= 0")
    if parsed["round_deadline_s"] < 0.0:
        raise ValueError("round-deadline-s doit etre >= 0")
    if parsed["fedstrag_enabled"] not in (0, 1):
        raise ValueError("fedstrag-enabled doit etre 0 ou 1")
    if parsed["fedstrag_alpha"] < 0.0:
        raise ValueError("fedstrag-alpha doit etre >= 0")
    if not (0.0 <= parsed["fedstrag_min_weight"] <= 1.0):
        raise ValueError("fedstrag-min-weight doit etre dans [0, 1]")
    if parsed["fedstrag_max_staleness"] < 0:
        raise ValueError("fedstrag-max-staleness doit etre >= 0")
    if parsed["downlink_comm_size_ratio"] < 0.0:
        raise ValueError("comm-size-ratio doit etre >= 0")
    if parsed["uplink_comm_size_ratio"] < 0.0:
        raise ValueError("ratio uplink effectif doit etre >= 0")
    if parsed["sim_model_mb"] < 0.0:
        raise ValueError("sim-model-mb doit etre >= 0")
    if parsed["momentum"] < 0.0:
        raise ValueError("momentum doit etre >= 0")
    if parsed["hardware_hetero"] not in (0, 1):
        raise ValueError("hardware-heterogeneity doit etre 0 ou 1")
    if not (0.0 < parsed["fraction_train"] <= 1.0):
        raise ValueError("fraction-train doit etre dans ]0, 1]")
    if parsed["compression_bits"] not in (4, 8, 16, 32):
        raise ValueError(
            "compression-quantization-bits doit etre 4, 8, 16 ou 32 "
            f"(recu {parsed['compression_bits']})"
        )
    if not (0.0 <= parsed["compression_sparsity"] < 1.0):
        raise ValueError(
            "compression-sparsity-ratio doit etre dans [0, 1[ "
            f"(recu {parsed['compression_sparsity']})"
        )


def print_banner(algo_name, cfg, banner_extra=""):
    """Affiche '[config] algo=... partitioning=... ...' au debut du run."""
    extra = f" alpha={cfg['dir_alpha']}" if cfg["partitioning"].lower() != "iid" else ""
    strag = (f" straggler-sim=1 deadline={cfg['round_deadline_s']}s"
             if cfg["straggler_sim"] else "")
    fedstrag = (
        f" fedstrag=1 alpha={cfg['fedstrag_alpha']}"
        f" min_w={cfg['fedstrag_min_weight']}"
        if cfg["fedstrag_enabled"] else ""
    )
    compr = (f" comm-size-ratio={cfg['comm_size_ratio_manual']}"
             if cfg["comm_size_ratio_manual"] != 1.0 else "")
    compression_tail = ""
    if cfg.get("compression_bits", 32) < 32:
        compression_tail += f" quant={cfg['compression_bits']}bits"
    if cfg.get("compression_sparsity", 0.0) > 0.0:
        compression_tail += f" sparse={cfg['compression_sparsity']}"
    if compression_tail:
        compression_tail += (
            f" (uplink_ratio={cfg.get('uplink_comm_size_ratio', 1.0):.4f})"
        )
    sim = f" sim-model-mb={cfg['sim_model_mb']}" if cfg["sim_model_mb"] > 0 else ""
    sd = f" seed={cfg['seed']}" if cfg["seed"] >= 0 else ""
    data_sd = f" data_seed={cfg['data_seed']}"
    mom = f" momentum={cfg['momentum']}"
    frac = (f" fraction_train={cfg['fraction_train']}"
            if cfg.get("fraction_train", 1.0) != 1.0 else "")
    hw = " hardware_heterogeneity=1" if cfg.get("hardware_hetero", 0) else ""
    print(f"[config] algo={algo_name}{banner_extra} partitioning={cfg['partitioning']}"
          f"{extra} num_rounds={cfg['num_rounds']} lr={cfg['lr']}{mom}"
          f"{strag}{fedstrag}{compr}{compression_tail}{sim}{sd}{data_sd}{frac}{hw}")


def server_evaluate(arrays, device, _cache=None, model_name="net",
                    eval_excluded_indices=None):
    """Eval du modele global sur le test set CIFAR-10 (10k images)."""
    if _cache is None:
        _cache = server_evaluate._cache
    excluded_key = tuple(sorted(int(i) for i in (eval_excluded_indices or [])))
    cache_key = ("loader", excluded_key)
    if cache_key not in _cache:
        ds = (
            get_testset_excluding_indices(excluded_key)
            if excluded_key else get_testset()
        )
        _cache[cache_key] = DataLoader(ds, batch_size=256, shuffle=False)
    net = get_model(model_name)
    net.load_state_dict(arrays.to_torch_state_dict())
    net.to(device)
    loss, acc, class_accs, mr, mf = test_with_class_accuracies(
        net, _cache[cache_key], device)
    return {
        "accuracy": float(acc), "loss": float(loss),
        "macro_recall": float(mr), "macro_f1": float(mf),
        "class_accs": [float(a) for a in class_accs],
    }


server_evaluate._cache = {}


def _client_metrics(rec):
    """Extrait un dict propre depuis un MetricRecord d'un client."""
    mr = next(iter(rec.metric_records.values()))
    energy_total = float(mr.get("energy_j", 0.0))
    energy_comp = float(mr.get("compute_energy_j", 0.0))
    energy_comm = float(mr.get("comm_energy_j", 0.0))
    return {
        "pid": int(mr.get("partition_id", -1)),
        "n": int(mr.get("num-examples", 0)),
        "epochs": float(mr.get("epochs_used", 0.0)),
        "tier": float(mr.get("resource_tier", 1.0)),
        "net_tier": int(mr.get("net_tier", 1)),
        "time": float(mr.get("local_time_s", 0.0)),
        "local_eval_time": float(mr.get("local_eval_time_s", 0.0)),
        "comm_time": float(mr.get("comm_time_s", 0.0)),
        "dropped": int(float(mr.get("dropped", 0.0)) >= 0.5),
        "energy": energy_total,
        "energy_compute": energy_comp,
        "energy_comm": energy_comm,
        "is_lan": int(float(mr.get("link_type_lan", 0.0)) >= 0.5),
        "battery_remaining_j": float(mr.get("battery_remaining_j", -1.0)),
        "battery_capacity_j": float(mr.get("battery_capacity_j", -1.0)),
        "battery_constrained": int(float(mr.get("battery_constrained", -1.0))),
        "hfl_edge_id": int(float(mr.get("hfl_edge_id", -1.0))),
        "local_loss": float(mr.get("local_eval_loss", 0.0)),
        "local_acc": float(mr.get("local_eval_acc", 0.0)),
        "cfl_width_ratio": float(mr.get("cfl_width_ratio", 1.0)),
        "fedstrag_late": int(float(mr.get("fedstrag_late", 0.0)) >= 0.5),
        "fedstrag_staleness": float(mr.get("fedstrag_staleness", 0.0)),
        "deadline_miss_s": float(mr.get("deadline_miss_s", 0.0)),
        "server_wait_time": float(
            mr.get("server_wait_time_s",
                   float(mr.get("local_time_s", 0.0))
                   + float(mr.get("comm_time_s", 0.0)))),
    }


def make_agg_train(round_info):
    """Cree le callback agg_train. round_info[0] sera rempli a chaque round."""
    def agg_train(records, wk):
        recs = list(records)
        details = [_client_metrics(rec) for rec in recs]

        # Les replies FedStrag late sont appliquees plus tard : on les exclut
        # des metriques du round courant.
        active = [
            d for d in details
            if not d["dropped"] and not d.get("fedstrag_late", 0) and d["n"] > 0
        ]
        weighted_recs = []
        for rec in recs:
            mr = next(iter(rec.metric_records.values()))
            is_late = float(mr.get("fedstrag_late", 0.0)) >= 0.5
            is_dropped = float(mr.get("dropped", 0.0)) >= 0.5
            if float(mr.get(wk, 0.0)) > 0.0 and not is_late and not is_dropped:
                weighted_recs.append(rec)
        m = aggregate_metricrecords(weighted_recs, wk) if weighted_recs else {}
        w = sum(d["n"] for d in active) or 1
        local_loss = sum(d["local_loss"] * d["n"] for d in active) / w
        local_acc = sum(d["local_acc"] * d["n"] for d in active) / w

        # Fairness inter-clients sur local_acc (les accuracies nulles sont
        # incluses volontairement).
        client_accs = [d["local_acc"] for d in active]
        if client_accs:
            from .metrics import jains_fairness_index as _jfi
            import numpy as _np
            jfi_clients_acc = _jfi(client_accs)
            client_acc_variance = float(_np.var(client_accs))
            worst_client_acc = float(min(client_accs))
            best_client_acc = float(max(client_accs))
            client_acc_gap = best_client_acc - worst_client_acc
        else:
            jfi_clients_acc = 0.0
            client_acc_variance = 0.0
            worst_client_acc = 0.0
            best_client_acc = 0.0
            client_acc_gap = 0.0

        # Compteur de participation appliquee ce round.
        part = {}
        for d in details:
            if (d["pid"] >= 0 and not d["dropped"]
                    and not d.get("fedstrag_late", 0) and d["n"] > 0):
                part[d["pid"]] = part.get(d["pid"], 0) + 1

        # Breakdown energie + width par tier hardware.
        tier_buckets = {0: [], 1: [], 2: []}
        for d in details:
            t = int(d.get("tier", 1))
            if t not in tier_buckets:
                tier_buckets[t] = []
            tier_buckets.setdefault(t, []).append(d)
        tier_energy = {}
        tier_count = {}
        tier_width = {}
        for t in (0, 1, 2):
            ds = tier_buckets.get(t, [])
            tier_energy[t] = sum(dd["energy"] for dd in ds)
            tier_count[t] = len(ds)
            widths_t = [float(dd.get("cfl_width_ratio", 1.0)) for dd in ds]
            tier_width[t] = (sum(widths_t) / len(widths_t)) if widths_t else 0.0

        round_info[0] = {
            "n_clients": len(details),
            "details": details,
            "n_dropped": sum(d["dropped"] for d in details),
            "n_active": len(active),
            "n_late": sum(int(d.get("fedstrag_late", 0)) for d in details),
            "participation_add": part,
            "energy_j_round": sum(d["energy"] for d in details),
            "tier0_energy_j_round": tier_energy[0],
            "tier1_energy_j_round": tier_energy[1],
            "tier2_energy_j_round": tier_energy[2],
            "tier0_n_clients": tier_count[0],
            "tier1_n_clients": tier_count[1],
            "tier2_n_clients": tier_count[2],
            "tier0_mean_width": tier_width[0],
            "tier1_mean_width": tier_width[1],
            "tier2_mean_width": tier_width[2],
            "compute_energy_j_round": sum(d["energy_compute"] for d in details),
            "comm_energy_j_round": sum(d["energy_comm"] for d in details),
            "local_loss": local_loss,
            "local_acc": local_acc,
            "mean_epochs": (sum(d["epochs"] for d in details) / len(details)
                            if details else 0.0),
            "mean_tier": (sum(d["tier"] for d in details) / len(details)
                          if details else 0.0),
            "times": [d["time"] for d in details],
            "comm_times": [d["comm_time"] for d in details],
            "jfi_clients_acc": jfi_clients_acc,
            "client_acc_variance": client_acc_variance,
            "client_acc_gap": client_acc_gap,
            "worst_client_acc": worst_client_acc,
            "best_client_acc": best_client_acc,
        }
        return m
    return agg_train


def empty_round_info():
    """Valeurs par defaut quand aucun client n'a repondu."""
    return {
        "n_clients": 0, "details": [], "n_dropped": 0, "n_late": 0,
        "n_active": 0,
        "participation_add": {}, "energy_j_round": 0.0,
        "compute_energy_j_round": 0.0, "comm_energy_j_round": 0.0,
        "local_loss": 0.0, "local_acc": 0.0,
        "mean_epochs": 0.0, "mean_tier": 0.0, "times": [], "comm_times": [],
        "jfi_clients_acc": 0.0, "client_acc_variance": 0.0,
        "client_acc_gap": 0.0, "worst_client_acc": 0.0, "best_client_acc": 0.0,
        "tier0_energy_j_round": 0.0, "tier1_energy_j_round": 0.0,
        "tier2_energy_j_round": 0.0,
        "tier0_n_clients": 0, "tier1_n_clients": 0, "tier2_n_clients": 0,
        "tier0_mean_width": 0.0, "tier1_mean_width": 0.0, "tier2_mean_width": 0.0,
    }


TIER_NAMES = {0: "weak", 1: "medium", 2: "strong"}
# Noms de tiers reseau selon le type de lien (WAN pour FedAvg/..., LAN pour HFL).
NET_NAMES_WAN = {0: "lora", 1: "lte", 2: "wifi"}
NET_NAMES_LAN = {0: "wlan", 1: "wlan+", 2: "eth"}
NET_NAMES = NET_NAMES_WAN  # backward-compat alias


def print_round(r, ev, tr, comm_mb, round_time_s, energy_round, energy_cumul,
                num_clients, straggler_sim, tail="", comm_lan_mb=0.0,
                technical_comm_mb=None):
    """Print detaille d'un round (par-client + resume).

    `comm_mb` = bytes WAN, `comm_lan_mb` = bytes LAN (HFL uniquement).
    """
    print(f"[round {r}] clients participants ({tr['n_clients']}):")
    for d in sorted(tr["details"], key=lambda x: x["pid"]):
        tname = TIER_NAMES.get(int(d["tier"]), "?")
        net_table = NET_NAMES_LAN if d.get("is_lan", 0) else NET_NAMES_WAN
        nname = net_table.get(d["net_tier"], "?")
        if d["dropped"]:
            flag = " DROP"
        elif d.get("fedstrag_late", 0):
            flag = f" LATE(s={d.get('fedstrag_staleness', 0.0):.0f})"
        else:
            flag = ""
        battery = ""
        if d["battery_capacity_j"] > 0:
            battery = (f"  bat={d['battery_remaining_j']:.1f}/"
                       f"{d['battery_capacity_j']:.1f}J")
        elif d["battery_constrained"] == 0:
            battery = "  bat=unlimited"
        edge = f"  edge=e{d['hfl_edge_id']}" if d["hfl_edge_id"] >= 0 else ""
        print(f"  pid={d['pid']:>2}  n={d['n']:>5}  E={d['epochs']:.0f}  "
              f"tier={tname:<6}  net={nname:<4}  t={d['time']:.2f}s  "
              f"comm={d['comm_time']:.2f}s  energy={d['energy']:.1f}J"
              f"{battery}{edge}{flag}")

    if straggler_sim:
        n_timeout = max(0, num_clients - tr["n_clients"])
        n_late = sum(int(d.get("fedstrag_late", 0)) for d in tr["details"])
        print(f"[round {r}] stragglers dropped={tr['n_dropped']}/{tr['n_clients']} "
              f"late={n_late}/{tr['n_clients']} timeout={n_timeout}/{num_clients}")

    gap = tr["local_acc"] - ev["accuracy"]
    mean_ct = sum(tr["times"]) / max(len(tr["times"]), 1)
    comm_ct = sum(tr.get("comm_times", [])) / max(len(tr.get("comm_times", [])), 1)
    print(f"[round {r}] server: acc={ev['accuracy']:.3f} loss={ev['loss']:.3f} "
          f"recall={ev['macro_recall']:.3f} f1={ev['macro_f1']:.3f}")
    print(f"[round {r}] local : acc={tr['local_acc']:.3f} loss={tr['local_loss']:.3f} "
          f"(gap_acc={gap:+.3f} = biais non-IID)")
    e_comp = tr.get("compute_energy_j_round", 0.0)
    e_comm = tr.get("comm_energy_j_round", 0.0)
    edge_e = tr.get("edge_cloud_energy_j_round", 0.0)
    edge_e_str = f" edge_cloud_E={edge_e:.1f}J" if edge_e > 0 else ""
    lan_str = f" lan={comm_lan_mb:.2f}MB" if comm_lan_mb > 0 else ""
    tech_str = ""
    if technical_comm_mb is not None:
        logical_total = comm_mb + comm_lan_mb
        if abs(float(technical_comm_mb) - logical_total) > 1e-6:
            tech_str = f" technical={technical_comm_mb:.2f}MB"
    print(f"[round {r}] comm={comm_mb:.2f}MB{lan_str}{tech_str} "
          f"n={tr['n_clients']} round={round_time_s:.1f}s "
          f"mean_ct={mean_ct:.2f}s "
          f"mean_commt={comm_ct:.2f}s "
          f"E={tr['mean_epochs']:.2f} tier={tr['mean_tier']:.2f} "
          f"energy={energy_round:.1f}J (compute={e_comp:.1f}J "
          f"comm={e_comm:.1f}J{edge_e_str} cumul={energy_cumul:.1f}J){tail}")


def finalize_run(arrays, accs_history, participation, total_time, eval_time,
                 wall_time, energy_cumul, cfg, project_dir_name,
                 strategy=None, extra_tail_fn=None,
                 energy_per_round=None, comm_per_round=None,
                 time_per_round=None, skip_server_eval=False):
    """Sauvegarde le modele final, ecrit metrics_summary.csv, copie les CSV."""
    log_summary(total_time, accs_history, participation,
                num_clients=cfg["num_clients"],
                energy_per_round=energy_per_round,
                comm_per_round=comm_per_round,
                time_per_round=time_per_round)
    log_participation(participation, num_clients=cfg["num_clients"])

    rtc = rounds_to_convergence(accs_history, ratio=0.9)
    r50 = rounds_to_target(accs_history, 0.5)
    r70 = rounds_to_target(accs_history, 0.7)
    r90 = rounds_to_target(accs_history, 0.9)
    final_acc = accs_history[-1] if accs_history else 0.0
    extra = f" alpha={cfg['dir_alpha']}" if cfg["partitioning"].lower() != "iid" else ""
    tail = extra_tail_fn(len(accs_history), strategy) if extra_tail_fn else ""

    print(f"[done] total_time={total_time:.1f}s (eval_time={eval_time:.1f}s "
          f"wall={wall_time:.1f}s) final_acc={final_acc:.3f} "
          f"rtc90={rtc} r50={r50} r70={r70} r90={r90} "
          f"energy_total={energy_cumul:.1f}J "
          f"partitioning={cfg['partitioning']}{extra}{tail}")

    # Sans modele global (skip_server_eval), on sauve les logits agreges si
    # la strategy les expose.
    if skip_server_eval:
        agg_logits = getattr(strategy, "_agg_logits", None)
        if agg_logits is not None:
            torch.save({"agg_logits": agg_logits},
                       os.path.join(ensure_dir(), "final_logits.pt"))
        else:
            print("[done] INFO: skip_server_eval=True mais strategy n'expose "
                  "pas `_agg_logits`. Skipping final save.")
    else:
        torch.save(arrays.to_torch_state_dict(),
                   os.path.join(ensure_dir(), "final_model.pt"))
    results_dir = get_results_dir()
    print(f"[done] CSV -> {results_dir}")

    dst = resolve_dst_results_dir(project_dir_name)
    if os.path.abspath(dst) != os.path.abspath(results_dir):
        try:
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(results_dir):
                shutil.copy2(os.path.join(results_dir, fn), os.path.join(dst, fn))
            print(f"[done] CSV copies dans {dst}")
        except Exception as e:
            print(f"[done] WARN copie CSV echouee: {e}")


def run_federated_training(
    grid, cfg, algo_name, strategy_class, strategy_kwargs,
    train_config_fn, project_dir_name,
    extra_tail_fn=None, banner_extra="",
    skip_server_eval=False,
):
    """Boucle FL complete partagee par les algos du repo.

    Le cout de communication est mesure a partir des bytes reellement
    transmis dans les messages (par FedAvgStrategy).

    skip_server_eval : si True, saute server_evaluate() (utile quand il n'y a
    pas de modele global a evaluer) ; l'accuracy "server" devient la moyenne
    ponderee des evaluations locales des clients.
    """
    logging.getLogger("flwr").setLevel(logging.WARNING)

    configure_energy_model(cfg)
    cfg = parse_config(cfg)
    set_results_dir(resolve_dst_results_dir(project_dir_name))
    reset_files()
    if cfg["seed"] >= 0:
        set_seed(cfg["seed"])  # init torch RNG cote serveur
    print_banner(algo_name, cfg, banner_extra)
    device = get_device()

    round_info = [None]  # rempli par agg_train a chaque round
    # La strategy applique elle-meme les ratios de compression par couche
    # (LAN/WAN) pour eviter une double application.
    scaffold_like = str(algo_name).strip().lower() == "scaffold"
    if scaffold_like and (
            cfg.get("compression_bits", 32) < 32
            or cfg.get("compression_sparsity", 0.0) > 0.0):
        print("[scaffold] WARN: compression quant/sparse ignoree pour SCAFFOLD "
              "afin de preserver les control variates.")
    strategy_kwargs = {
        **strategy_kwargs,
        "comm_size_ratio": cfg["downlink_comm_size_ratio"],
        "uplink_size_ratio": (
            cfg["comm_size_ratio_manual"] if scaffold_like
            else cfg["uplink_comm_size_ratio"]
        ),
        "sim_model_mb": cfg["sim_model_mb"],
        "model_name": cfg["model_name"],
    }
    strategy = strategy_class(
        fraction_train=float(cfg["fraction_train"]),
        fraction_evaluate=0.0,                    # eval = serveur uniquement
        train_metrics_aggr_fn=make_agg_train(round_info),
        **strategy_kwargs,
    )
    accepts_late_stragglers = bool(
        getattr(strategy, "accepts_late_stragglers", False))
    start_kwargs = {"grid": grid, "num_rounds": 1}
    if (cfg["straggler_sim"] and cfg["round_deadline_s"] > 0
            and not accepts_late_stragglers):
        start_kwargs["timeout"] = cfg["round_deadline_s"]

    arrays = ArrayRecord(get_model(cfg["model_name"]).state_dict())
    accs_history, participation = [], {}
    fl_time_total, eval_time_total, energy_cumul = 0.0, 0.0, 0.0
    compute_energy_cumul, comm_energy_cumul = 0.0, 0.0
    edge_cloud_energy_cumul = 0.0
    eval_excluded_indices = None
    if (
        str(project_dir_name).lower() in ("fedavg_kl", "edgefd")
        and cfg.get("fedavgkl_eval_exclude_proxy", 1)
        and int(cfg.get("fedavgkl_proxy_size", 0)) > 0
    ):
        _, eval_excluded_indices = get_proxy_subset(
            proxy_size=int(cfg["fedavgkl_proxy_size"]),
            seed=int(cfg.get("data_seed", 42)),
        )
        print(
            "[FedAvg+KL] INFO: evaluation serveur sur CIFAR-10 test hors proxy "
            f"({len(eval_excluded_indices)} images exclues)."
        )
    # Historiques par round pour energy/comm/time-to-target du summary final.
    energy_per_round_hist = []
    comm_per_round_hist = []
    time_per_round_hist = []
    best_acc, no_improve = 0.0, 0
    t_start = time.perf_counter()

    for r in range(1, cfg["num_rounds"] + 1):
        # 1) Un round FL (clients trainent)
        start_kwargs["initial_arrays"] = arrays
        start_kwargs["train_config"] = ConfigRecord(train_config_fn(r, cfg["lr"], cfg))
        t0 = time.perf_counter()
        result = strategy.start(**start_kwargs)
        round_wall_s = time.perf_counter() - t0
        if result.arrays is not None:
            arrays = result.arrays

        # 2) Eval cote serveur (temps mesure separement)
        if skip_server_eval:
            tr_preview = round_info[0] or empty_round_info()
            ev = {
                "accuracy": float(tr_preview.get("local_acc", 0.0)),
                "loss": float(tr_preview.get("local_loss", 0.0)),
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "class_accs": [0.0] * 10,
            }
        else:
            t_eval = time.perf_counter()
            ev = server_evaluate(
                arrays, device, model_name=cfg["model_name"],
                eval_excluded_indices=eval_excluded_indices)
            eval_time_total += time.perf_counter() - t_eval
        accs_history.append(ev["accuracy"])

        # 3) Recup metriques clients + maj etat
        tr = round_info[0] or empty_round_info()
        client_energy_round = tr["energy_j_round"]
        edge_cloud_energy_round = float(
            getattr(strategy, "last_edge_cloud_energy_j", 0.0))
        tr["client_energy_j_round"] = client_energy_round
        tr["edge_cloud_energy_j_round"] = edge_cloud_energy_round
        if edge_cloud_energy_round:
            tr["energy_j_round"] = client_energy_round + edge_cloud_energy_round
            tr["comm_energy_j_round"] = (
                tr.get("comm_energy_j_round", 0.0) + edge_cloud_energy_round)

        # Temps de round : max(wall-clock, temps simule client) pour ne pas
        # sous-estimer quand une partie de la communication n'est pas
        # materialisee par sleep().
        client_sim_time_s = max(
            (d.get("server_wait_time", d["time"] + d["comm_time"])
             for d in tr["details"]),
            default=0.0,
        )
        extra_round_time_s = float(getattr(strategy, "last_extra_round_time_s", 0.0))
        if accepts_late_stragglers:
            # FedStrag : le temps logique du round reste borne par la deadline.
            round_time_s = client_sim_time_s + extra_round_time_s
        else:
            round_time_s = max(round_wall_s, client_sim_time_s) + extra_round_time_s
        fl_time_total += round_time_s

        # Si la strategy suit la batterie cote serveur, on remplace les
        # sentinels -1 dans les details.
        bat_map = getattr(strategy, "last_battery_remaining_j", None)
        if bat_map:
            for d in tr["details"]:
                pid = int(d.get("pid", -1))
                if pid in bat_map:
                    d["battery_remaining_j"] = float(bat_map[pid])

        for pid, n in tr["participation_add"].items():
            participation[pid] = participation.get(pid, 0) + n
        for pid in getattr(strategy, "last_fedstrag_applied_pids", []):
            participation[pid] = participation.get(pid, 0) + 1
        energy_cumul += tr["energy_j_round"]
        compute_energy_cumul += tr.get("compute_energy_j_round", 0.0)
        comm_energy_cumul += tr.get("comm_energy_j_round", 0.0)
        edge_cloud_energy_cumul += edge_cloud_energy_round
        energy_per_round_hist.append(float(tr["energy_j_round"]))
        time_per_round_hist.append(float(round_time_s))
        # comm_mb = cout WAN du round ; les ratios sont deja appliques par
        # la strategy. Pour HFL, last_lan_*_bytes couvre les clients <-> edges.
        wan_bytes = strategy.last_downlink_bytes + strategy.last_uplink_bytes
        lan_bytes = (getattr(strategy, "last_lan_downlink_bytes", 0)
                     + getattr(strategy, "last_lan_uplink_bytes", 0))
        technical_bytes = (
            getattr(strategy, "last_technical_downlink_bytes", wan_bytes)
            + getattr(strategy, "last_technical_uplink_bytes", 0)
        )
        comm_mb = wan_bytes / (1024.0 * 1024.0)
        comm_lan_mb = lan_bytes / (1024.0 * 1024.0)
        technical_comm_mb = technical_bytes / (1024.0 * 1024.0)
        comm_per_round_hist.append(comm_mb + comm_lan_mb)

        comm_times = tr.get("comm_times", [])
        mean_comm = sum(comm_times) / max(len(comm_times), 1)
        max_comm = max(comm_times, default=0.0)
        edge_cloud_time_s = float(getattr(strategy, "last_edge_cloud_time_s", 0.0))
        edge_cloud_downlink_time_s = float(
            getattr(strategy, "last_edge_cloud_downlink_time_s", 0.0))
        edge_cloud_uplink_time_s = float(
            getattr(strategy, "last_edge_cloud_uplink_time_s", 0.0))
        fedstrag_buffered = int(getattr(strategy, "last_fedstrag_buffered", 0))
        fedstrag_applied = int(getattr(strategy, "last_fedstrag_applied", 0))
        fedstrag_pending = int(len(getattr(strategy, "_fedstrag_buffer", [])))
        # Diagnostics algo-specifiques (getattr renvoie 0.0 pour les autres algos).
        fedeve_G = float(getattr(strategy, "last_kalman_gain", 0.0))
        fedeve_sQ = float(getattr(strategy, "last_sigma_Q", 0.0))
        fedeve_sR = float(getattr(strategy, "last_sigma_R", 0.0))

        cfl_widths = list(getattr(strategy, "last_widths", []) or [])
        cfl_mean_width = (sum(cfl_widths) / len(cfl_widths)) if cfl_widths else 0.0
        cfl_n_active_widths = len(set(cfl_widths))
        cfl_pred_loss = float(
            getattr(getattr(strategy, "predictor_trainer", None),
                    "last_loss", 0.0))
        scaffold_cg = float(getattr(strategy, "last_c_global_norm", 0.0))
        scaffold_dc = float(getattr(strategy, "last_delta_c_norm", 0.0))
        fednova_tau_eff = float(getattr(strategy, "last_tau_eff", 0.0))
        fednova_tau_min = float(getattr(strategy, "last_tau_min", 0.0))
        fednova_tau_max = float(getattr(strategy, "last_tau_max", 0.0))
        fedprox_prox = float(getattr(strategy, "last_prox_term_mean", 0.0))
        hfl_bat_min = float(getattr(strategy, "last_battery_min_ratio", 0.0))
        hfl_bat_mean = float(getattr(strategy, "last_battery_mean_ratio", 0.0))
        hfl_q_mean = float(getattr(strategy, "last_queue_mean", 0.0))
        hfl_q_max = float(getattr(strategy, "last_queue_max", 0.0))

        # 4) Log CSV
        log_round(
            r, ev["accuracy"], ev["loss"], ev["macro_recall"], ev["macro_f1"],
            ev["class_accs"],
            comm_cost_mb=comm_mb,
            comm_lan_mb=comm_lan_mb,
            technical_comm_mb=technical_comm_mb,
            round_time_s=round_time_s,
            mean_client_time_s=sum(tr["times"]) / max(len(tr["times"]), 1),
            max_client_time_s=max(tr["times"], default=0.0),
            mean_comm_time_s=mean_comm,
            max_comm_time_s=max_comm,
            edge_cloud_time_s=edge_cloud_time_s,
            edge_cloud_downlink_time_s=edge_cloud_downlink_time_s,
            edge_cloud_uplink_time_s=edge_cloud_uplink_time_s,
            mean_epochs_used=tr["mean_epochs"],
            mean_resource_tier=tr["mean_tier"],
            energy_j_round=tr["energy_j_round"],
            energy_j_cumulative=energy_cumul,
            client_energy_j_round=tr.get("client_energy_j_round", tr["energy_j_round"]),
            edge_cloud_energy_j_round=tr.get("edge_cloud_energy_j_round", 0.0),
            edge_cloud_energy_j_cumulative=edge_cloud_energy_cumul,
            compute_energy_j_round=tr.get("compute_energy_j_round", 0.0),
            comm_energy_j_round=tr.get("comm_energy_j_round", 0.0),
            compute_energy_j_cumulative=compute_energy_cumul,
            comm_energy_j_cumulative=comm_energy_cumul,
            local_loss=tr["local_loss"],
            local_acc=tr["local_acc"],
            fedstrag_buffered=fedstrag_buffered,
            fedstrag_applied=fedstrag_applied,
            fedstrag_pending=fedstrag_pending,
            fedeve_kalman_gain=fedeve_G,
            fedeve_sigma_Q=fedeve_sQ,
            fedeve_sigma_R=fedeve_sR,
            jfi_clients_acc=tr.get("jfi_clients_acc", 0.0),
            client_acc_variance=tr.get("client_acc_variance", 0.0),
            client_acc_gap=tr.get("client_acc_gap", 0.0),
            worst_client_acc=tr.get("worst_client_acc", 0.0),
            best_client_acc=tr.get("best_client_acc", 0.0),
            cfl_mean_width=cfl_mean_width,
            cfl_predictor_loss=cfl_pred_loss,
            cfl_n_active_widths=cfl_n_active_widths,
            scaffold_cg_norm=scaffold_cg,
            scaffold_dc_norm=scaffold_dc,
            fednova_tau_eff=fednova_tau_eff,
            fednova_tau_min=fednova_tau_min,
            fednova_tau_max=fednova_tau_max,
            fedprox_prox_term=fedprox_prox,
            hfl_battery_min_ratio=hfl_bat_min,
            hfl_battery_mean_ratio=hfl_bat_mean,
            hfl_queue_mean=hfl_q_mean,
            hfl_queue_max=hfl_q_max,
            tier0_energy_j_round=tr.get("tier0_energy_j_round", 0.0),
            tier1_energy_j_round=tr.get("tier1_energy_j_round", 0.0),
            tier2_energy_j_round=tr.get("tier2_energy_j_round", 0.0),
            tier0_n_clients=tr.get("tier0_n_clients", 0),
            tier1_n_clients=tr.get("tier1_n_clients", 0),
            tier2_n_clients=tr.get("tier2_n_clients", 0),
            tier0_mean_width=tr.get("tier0_mean_width", 0.0),
            tier1_mean_width=tr.get("tier1_mean_width", 0.0),
            tier2_mean_width=tr.get("tier2_mean_width", 0.0),
        )
        log_client_details(r, tr["details"])

        # 5) Print
        tail = extra_tail_fn(r, strategy) if extra_tail_fn else ""
        print_round(r, ev, tr, comm_mb, round_time_s,
                    tr["energy_j_round"], energy_cumul,
                    cfg["num_clients"], cfg["straggler_sim"], tail,
                    comm_lan_mb=comm_lan_mb,
                    technical_comm_mb=technical_comm_mb)

        # 6) Early stopping
        if cfg["es_patience"] > 0:
            if ev["accuracy"] > best_acc + cfg["es_min_delta"]:
                best_acc, no_improve = ev["accuracy"], 0
            else:
                no_improve += 1
                if no_improve >= cfg["es_patience"]:
                    print(f"[early-stop] convergence a r={r} "
                          f"(best_acc={best_acc:.3f}, patience={cfg['es_patience']})")
                    break

        # Hook optionnel : les strategies battery-aware peuvent raccourcir
        # leur horizon quand l'early stopping approche.
        if hasattr(strategy, "notify_es_progress"):
            try:
                strategy.notify_es_progress(no_improve, cfg["es_patience"])
            except (TypeError, ValueError, AttributeError) as exc:
                import traceback as _tb
                print(f"[warn] notify_es_progress signature/contrat invalide "
                      f"({type(exc).__name__}: {exc})")
                _tb.print_exc()

        round_info[0] = None  # reset pour le prochain round

    finalize_run(
        arrays, accs_history, participation,
        total_time=fl_time_total,
        eval_time=eval_time_total,
        wall_time=time.perf_counter() - t_start,
        energy_cumul=energy_cumul,
        cfg=cfg, project_dir_name=project_dir_name,
        strategy=strategy, extra_tail_fn=extra_tail_fn,
        energy_per_round=energy_per_round_hist,
        comm_per_round=comm_per_round_hist,
        time_per_round=time_per_round_hist,
        skip_server_eval=skip_server_eval,
    )
