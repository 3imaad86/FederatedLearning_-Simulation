"""ServerApp HFL-BEFL + CFL.

Glue : lit la config, valide, instancie CFLHFLBeflStrategy et appelle
run_federated_training.
"""

from ._local_imports import ensure_workspace_packages

ensure_workspace_packages()

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training

from .strategy import CFLHFLBeflStrategy

app = ServerApp()


def _hfl_befl_cfl_tail(_, strategy):
    """Tail enrichi : HFL-BEFL + CFL diagnostics."""
    # HFL-BEFL classique
    loads = ",".join(
        f"e{edge}:{count}"
        for edge, count in sorted(strategy.last_edge_loads.items())
    )
    edge_time = getattr(strategy, "last_edge_cloud_time_s", 0.0)
    cloud_sync = getattr(strategy, "last_cloud_sync", 0)
    local_steps = getattr(strategy, "edge_local_steps", 1)
    rl = getattr(strategy, "rl_enabled", 0)
    rl_tail = f" rl={rl}"
    if rl:
        baseline = getattr(strategy, "_reward_baseline", 0.0)
        rl_tail += f" rl_baseline={baseline:+.3f}"
    # CFL : distribution widths + sante predictor
    dist = getattr(strategy, "last_width_dist", {}) or {}
    widths_str = ",".join(f"{w:.2f}:{n}" for w, n in sorted(dist.items()))
    trainer = getattr(strategy, "predictor_trainer", None)
    pred_loss = float(getattr(trainer, "last_loss", 0.0))
    cfl_tail = f" widths=[{widths_str}] pred_loss={pred_loss:.4f}"
    # BEFL diagnostics (batterie + queue)
    bat_mean = float(getattr(strategy, "last_battery_mean_ratio", 1.0))
    bat_tail = ""
    if bat_mean < 1.0:
        bat_min = float(getattr(strategy, "last_battery_min_ratio", 1.0))
        q_max = float(getattr(strategy, "last_queue_max", 0.0))
        bat_tail = (f" bat=mean{bat_mean:.2f}/min{bat_min:.2f}"
                    f" Q_max={q_max:.1f}")
    return (f" edges=[{loads}] k1={local_steps} sync={cloud_sync}"
            f" edge_cloud={edge_time:.2f}s"
            f"{rl_tail}{cfl_tail}{bat_tail}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    # HFL/BEFL/RL params
    num_edges = int(cfg.get("hfl-num-edges", 3))
    edge_cloud_ratio = float(cfg.get("hfl-edge-cloud-ratio", 1.0))
    edge_cloud_bw = float(cfg.get("hfl-edge-cloud-bw-mbps", 5.0))
    edge_cloud_rtt = float(cfg.get("hfl-edge-cloud-rtt-s", 0.5))
    edge_cloud_jitter = float(cfg.get("hfl-edge-cloud-jitter-s", 0.0))
    edge_cloud_deadline = float(cfg.get("hfl-edge-cloud-deadline-s", 0.0))
    edge_local_steps = int(cfg.get("hfl-local-steps", 3))
    straggler_sim = int(cfg.get("straggler-sim", 0))
    num_clients = int(cfg.get("num-clients", 10))
    rl_enabled = int(cfg.get("hfl-rl-edge-assignment", 0))
    rl_epsilon = float(cfg.get("hfl-rl-epsilon", 0.2))
    rl_lr = float(cfg.get("hfl-rl-lr", 0.01))
    total_rounds = int(cfg.get("num-server-rounds", 30))
    befl_V = float(cfg.get("befl-V", 100.0))
    befl_base_epochs = int(cfg.get("local-epochs", 2))
    befl_death_threshold = float(cfg.get("befl-death-threshold", 0.05))
    fedstrag_enabled = int(cfg.get("fedstrag-enabled", 0))
    fedstrag_alpha = float(cfg.get("fedstrag-alpha", 1.0))
    fedstrag_min_weight = float(cfg.get("fedstrag-min-weight", 0.05))
    fedstrag_max_staleness = int(cfg.get("fedstrag-max-staleness", 0))
    es_horizon_trigger = int(cfg.get("befl-es-horizon-trigger", 3))
    rl_seed = int(cfg.get("seed", -1))

    # CFL params
    cfl_search_times = int(cfg.get("cfl-search-times", 10))
    cfl_predictor_lr = float(cfg.get("cfl-predictor-lr", 0.01))
    cfl_predictor_hidden = int(cfg.get("cfl-predictor-hidden", 32))
    raw_widths = cfg.get("cfl-candidate-widths", "0.25,0.5,0.75,1.0")
    if isinstance(raw_widths, str):
        cfl_candidate_widths = [
            float(x.strip()) for x in raw_widths.split(",") if x.strip()
        ]
    else:
        cfl_candidate_widths = [float(x) for x in raw_widths]

    # validations
    if num_edges < 1:
        raise ValueError("hfl-num-edges doit etre >= 1")
    if edge_local_steps < 1:
        raise ValueError("hfl-local-steps doit etre >= 1")
    if edge_cloud_ratio < 0.0:
        raise ValueError("hfl-edge-cloud-ratio doit etre >= 0")
    if edge_cloud_bw <= 0.0:
        raise ValueError("hfl-edge-cloud-bw-mbps doit etre > 0")
    if edge_cloud_rtt < 0.0:
        raise ValueError("hfl-edge-cloud-rtt-s doit etre >= 0")
    if edge_cloud_jitter < 0.0:
        raise ValueError("hfl-edge-cloud-jitter-s doit etre >= 0")
    if edge_cloud_deadline < 0.0:
        raise ValueError("hfl-edge-cloud-deadline-s doit etre >= 0")
    if rl_enabled not in (0, 1):
        raise ValueError("hfl-rl-edge-assignment doit etre 0 ou 1")
    if not (0.0 <= rl_epsilon <= 1.0):
        raise ValueError("hfl-rl-epsilon doit etre dans [0, 1]")
    if rl_lr <= 0.0:
        raise ValueError("hfl-rl-lr doit etre > 0")
    if befl_V < 0.0:
        raise ValueError("befl-V doit etre >= 0")
    if not (0.0 <= befl_death_threshold <= 1.0):
        raise ValueError("befl-death-threshold doit etre dans [0, 1]")
    if fedstrag_enabled not in (0, 1):
        raise ValueError("fedstrag-enabled doit etre 0 ou 1")
    if fedstrag_alpha < 0.0:
        raise ValueError("fedstrag-alpha doit etre >= 0")
    if not (0.0 <= fedstrag_min_weight <= 1.0):
        raise ValueError("fedstrag-min-weight doit etre dans [0, 1]")
    if fedstrag_max_staleness < 0:
        raise ValueError("fedstrag-max-staleness doit etre >= 0")
    if cfl_search_times < 1:
        raise ValueError("cfl-search-times doit etre >= 1")
    if cfl_predictor_lr <= 0.0:
        raise ValueError("cfl-predictor-lr doit etre > 0")
    if cfl_predictor_hidden < 4:
        raise ValueError("cfl-predictor-hidden doit etre >= 4")
    if not cfl_candidate_widths:
        raise ValueError("cfl-candidate-widths doit etre non vide")
    if any(w <= 0.0 or w > 1.0 for w in cfl_candidate_widths):
        raise ValueError(
            "cfl-candidate-widths doit avoir des valeurs dans ]0, 1]")
    if fedstrag_enabled and not straggler_sim:
        print(
            "[HFL-BEFL-CFL] WARN: fedstrag-enabled=1 mais straggler-sim=0 : "
            "aucun reply late ne sera genere."
        )
    if fedstrag_enabled:
        print(
            "[HFL-BEFL-CFL] WARN: fedstrag-enabled=1 combine a CFL. "
            "La combo n'est pas formellement supportee : "
            "les replies late ont un width potentiellement different du "
            "round courant -> agregation edge degradee. "
            "Recommande : fedstrag-enabled=0."
        )

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="HFL-BEFL-CFL",
        strategy_class=CFLHFLBeflStrategy,
        strategy_kwargs={
            # HFL-BEFL inherited args
            "num_edges": num_edges,
            "num_clients": num_clients,
            "edge_cloud_ratio": edge_cloud_ratio,
            "edge_cloud_bw_mbps": edge_cloud_bw,
            "edge_cloud_rtt_s": edge_cloud_rtt,
            "edge_cloud_jitter_s": edge_cloud_jitter,
            "edge_cloud_deadline_s": edge_cloud_deadline,
            "edge_local_steps": edge_local_steps,
            "straggler_sim": straggler_sim,
            "rl_enabled": rl_enabled,
            "rl_epsilon": rl_epsilon,
            "rl_lr": rl_lr,
            "total_rounds": total_rounds,
            "befl_V": befl_V,
            "befl_base_epochs": befl_base_epochs,
            "befl_death_threshold": befl_death_threshold,
            "seed": rl_seed,
            "fedstrag_enabled": fedstrag_enabled,
            "fedstrag_alpha": fedstrag_alpha,
            "fedstrag_min_weight": fedstrag_min_weight,
            "fedstrag_max_staleness": fedstrag_max_staleness,
            "es_horizon_trigger": es_horizon_trigger,
            # CFL specific
            "cfl_search_times": cfl_search_times,
            "cfl_predictor_lr": cfl_predictor_lr,
            "cfl_predictor_hidden": cfl_predictor_hidden,
            "cfl_candidate_widths": cfl_candidate_widths,
        },
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="hfl_befl_cfl",
        extra_tail_fn=_hfl_befl_cfl_tail,
        banner_extra=(
            f" edges={num_edges} k1={edge_local_steps}"
            f" rl={rl_enabled} cfl_widths={cfl_candidate_widths}"
            f" S={cfl_search_times}"
        ),
    )
