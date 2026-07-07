"""ServerApp FairFed (Ezzeldin et al. 2023, adapte CIFAR-10).

Algorithme :
    p_k         = n_k / sum(n)              (poids FedAvg de base)
    F_global    = sum(p_k * F_k)            (consensus de loss)
    Delta_k     = |F_k - F_global|          (deviation par rapport au consensus)
    mean_delta  = mean(Delta_k)
    w_k         = max(0, p_k - beta * (Delta_k - mean_delta))
                  (clients pres du consensus -> upweight, outliers -> downweight)
    Normalise w_k puis aggrege.

Difference avec q-FedAvg :
    q-FedAvg : upweight les clients haute loss (aide les strugglers)
    FairFed  : upweight les clients pres du consensus (filtre les outliers)
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training

from fairfed.strategy import FairFedStrategy

app = ServerApp()


def _fairfed_tail(_, strategy):
    """Diagnostics FairFed par round.

    F_global   : consensus de loss (sum p_k * F_k).
    F_min/max  : extreme des F_k -> juge la dispersion inter-clients.
    w_var      : variance des poids FairFed normalises. Plus c'est haut,
                 plus le serveur upweight/downweight, plus FairFed agit.
                 w_var ~ 0 = FairFed = FedAvg ce round.
    """
    f_g = float(getattr(strategy, "last_fairfed_F_global", 0.0))
    f_min = float(getattr(strategy, "last_fairfed_F_min", 0.0))
    f_max = float(getattr(strategy, "last_fairfed_F_max", 0.0))
    w_var = float(getattr(strategy, "last_fairfed_w_var", 0.0))
    return (f" F_g={f_g:.3f} [{f_min:.3f}, {f_max:.3f}]"
            f" w_var={w_var:.2e}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    beta = float(cfg.get("fairfed-beta", 0.1))
    # EMA inter-rounds des poids (Ezzeldin 2023). 1.0 = zero-shot, 0.5 =
    # compromis recommande par le papier, stable sous non-IID extreme.
    ema_alpha = float(cfg.get("fairfed-ema-alpha", 0.5))
    max_shift_ratio = float(cfg.get("fairfed-max-shift-ratio", 0.5))
    if beta < 0.0:
        raise ValueError(f"FairFed exige beta >= 0 (recu beta={beta})")
    if not (0.0 < ema_alpha <= 1.0):
        raise ValueError(
            f"FairFed exige 0 < fairfed-ema-alpha <= 1 "
            f"(recu {ema_alpha})"
        )
    if max_shift_ratio < 0.0:
        raise ValueError(
            "FairFed exige fairfed-max-shift-ratio >= 0 "
            f"(recu {max_shift_ratio})"
        )

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FairFed",
        strategy_class=FairFedStrategy,
        strategy_kwargs={
            "beta": beta,
            "ema_alpha": ema_alpha,
            "max_shift_ratio": max_shift_ratio,
        },
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fairfed",
        banner_extra=(
            f" beta={beta} ema={ema_alpha}"
            f" max_shift={max_shift_ratio}"
        ),
        extra_tail_fn=_fairfed_tail,
    )
