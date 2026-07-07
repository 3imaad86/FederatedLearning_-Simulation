"""ServerApp q-FedAvg (Li, Sanjabi, Smith 2019).

q grand -> les clients avec haute loss contribuent plus (fairness) ;
q = 0 -> proche d'une moyenne uniforme des updates.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training

from qfedavg.strategy import QFedAvgStrategy

app = ServerApp()


def _qfedavg_tail(_, strategy):
    """Ajoute au log par round les losses F (min/max/mean) et |step|."""
    f_mean = float(getattr(strategy, "last_qfedavg_F_mean", 0.0))
    f_min = float(getattr(strategy, "last_qfedavg_F_min", 0.0))
    f_max = float(getattr(strategy, "last_qfedavg_F_max", 0.0))
    step = float(getattr(strategy, "last_qfedavg_step_norm", 0.0))
    return (f" F={f_mean:.3f} [{f_min:.3f}, {f_max:.3f}]"
            f" |step|={step:.2e}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    q = float(cfg.get("qfedavg-q", 1.0))
    lr = float(cfg.get("learning-rate", 0.01))
    L_cfg = float(cfg.get("qfedavg-L", 0.0))
    local_epochs = int(cfg.get("local-epochs", 2))
    epochs_hetero = int(cfg.get("epochs-heterogeneity", 0))
    if q < 0.0:
        raise ValueError(f"q-FedAvg exige q >= 0 (recu q={q})")
    if lr <= 0.0:
        raise ValueError(f"q-FedAvg exige learning-rate > 0 (recu lr={lr})")

    # qfedavg-L=0 signifie auto : la recommandation pratique q-FedAvg
    # est L ~= 1 / lr_local.
    auto_L = L_cfg <= 0.0
    L = (1.0 / lr) if auto_L else L_cfg

    # Delta_k = L*(w - w_local) n'est un bon proxy du gradient que pour peu
    # de pas SGD : avec E > 2, ||Delta_k||^2 explose et l'update collapse.
    risky_E = (local_epochs > 2) or epochs_hetero
    risky_L = L >= 50.0
    if risky_E and risky_L and auto_L:
        print(
            f"[q-FedAvg] WARN: L=1/lr={L:.1f} avec "
            f"{'epochs-heterogeneity=1' if epochs_hetero else f'local-epochs={local_epochs}'}"
            f" -> risque d'effective-LR collapse (||Delta||^2 domine h_k). "
            f"Si l'accuracy serveur stagne, essayer:\n"
            f"  (a) qfedavg-L=1.0 (override manuel), OU\n"
            f"  (b) local-epochs=1 epochs-heterogeneity=0 (regime du papier)."
        )

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="q-FedAvg",
        strategy_class=QFedAvgStrategy,
        strategy_kwargs={"q": q, "L": L},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="qfedavg",
        banner_extra=f" q={q} L={L:.6g}{' (auto=1/lr)' if auto_L else ''}",
        extra_tail_fn=_qfedavg_tail,
    )
