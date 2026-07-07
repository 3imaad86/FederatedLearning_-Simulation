"""ServerApp FedEve (Shen et al. 2025) : fusion Kalman entre le momentum
serveur et la moyenne des updates clients. Un seul hyperparametre (eta_g),
le reste est appris depuis les variances observees.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedEveStrategy

app = ServerApp()


def _fedeve_tail(_, strategy):
    """Ajoute au log par round le gain de Kalman et les variances sQ/sR."""
    G = float(getattr(strategy, "last_kalman_gain", 0.0))
    sQ = float(getattr(strategy, "last_sigma_Q", 0.0))
    sR = float(getattr(strategy, "last_sigma_R", 0.0))
    return f" kal={G:.3f} sQ={sQ:.2e} sR={sR:.2e}"


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    eta_g = float(cfg.get("fedeve-server-lr", 1.0))
    if eta_g <= 0.0:
        raise ValueError(f"FedEve exige fedeve-server-lr > 0 (recu {eta_g})")

    # Heuristiques hors papier (plancher G_kal + bootstrap cold-start) pour
    # la robustesse sous non-IID extreme.
    kalman_gain_min = float(cfg.get("fedeve-kalman-gain-min", 0.05))
    cold_start_bootstrap = int(cfg.get("fedeve-cold-start-bootstrap", 1))
    if not (0.0 <= kalman_gain_min < 1.0):
        raise ValueError(
            f"fedeve-kalman-gain-min doit etre dans [0, 1[ "
            f"(recu {kalman_gain_min})"
        )
    if cold_start_bootstrap not in (0, 1):
        raise ValueError(
            "fedeve-cold-start-bootstrap doit etre 0 ou 1 "
            f"(recu {cold_start_bootstrap})"
        )
    paper_strict = (kalman_gain_min == 0.0 and cold_start_bootstrap == 0)
    if not paper_strict:
        print(
            f"[fedeve] INFO: mode 'robuste' active "
            f"(kalman_gain_min={kalman_gain_min}, "
            f"cold_start_bootstrap={bool(cold_start_bootstrap)}). "
            "Pour reproduire Shen et al. 2025 strictement, mettre "
            "`fedeve-kalman-gain-min=0.0` et `fedeve-cold-start-bootstrap=0`."
        )

    # La theorie FedEve suppose E=1 ; on previent si ce n'est pas le cas.
    local_epochs = int(cfg.get("local-epochs", 1))
    epochs_hetero = int(cfg.get("epochs-heterogeneity", 0))
    if local_epochs > 1 or epochs_hetero:
        print(
            f"[fedeve] WARN: local-epochs={local_epochs} "
            f"epochs-heterogeneity={epochs_hetero}. "
            "Le papier (Shen et al. 2025, Sec 3.1) utilise E=1. "
            "Multi-epoch supporte mais l'estimation des variances Q_t/R_t "
            "peut etre moins fiable -> regarde `kal` dans les logs pour "
            "diagnostiquer."
        )

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FedEve",
        strategy_class=FedEveStrategy,
        strategy_kwargs={
            "server_lr": eta_g,
            "kalman_gain_min": kalman_gain_min,
            "cold_start_bootstrap": bool(cold_start_bootstrap),
        },
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fedeve",
        banner_extra=(
            f" eta_g={eta_g}"
            f" G_min={kalman_gain_min}"
            f" bootstrap={'paper-strict' if paper_strict else 'robust'}"
        ),
        extra_tail_fn=_fedeve_tail,
    )
