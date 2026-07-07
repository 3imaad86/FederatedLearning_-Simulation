"""ServerApp FedAvg+KL : agregation FedAvg + distillation KL via dataset proxy.

Variante hybride (FedAvg + regularisation KL) inspiree de la partie
proxy/KMeans-DRE d'EdgeFD (Liu et al. 2025).
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training

from fedavg_kl.strategy import FedAvgKLStrategy

app = ServerApp()


def _fedavgkl_tail(_, strategy):
    """Ajoute au log par round : id_ratio, logits actifs et surcout comm des logits."""
    id_ratio = float(getattr(strategy, "last_id_ratio", 0.0))
    n_active = int(getattr(strategy, "last_logits_active", 0))
    logits_bytes = int(getattr(strategy, "last_logits_bytes", 0))
    logits_mb = logits_bytes / (1024.0 * 1024.0)
    return (f" id_ratio={id_ratio:.2f} logits_active={n_active}"
            f" logits_MB={logits_mb:.3f}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    proxy_size = int(cfg.get("fedavgkl-proxy-size", 2000))
    proxy_per_round = int(cfg.get("fedavgkl-proxy-per-round", 500))
    distill_lambda = float(cfg.get("fedavgkl-distill-lambda", 0.5))
    distill_T = float(cfg.get("fedavgkl-distill-temperature", 4.0))

    if proxy_size < 1:
        raise ValueError(f"fedavgkl-proxy-size doit etre >= 1 (recu {proxy_size})")
    if proxy_per_round < 0:
        raise ValueError(
            f"fedavgkl-proxy-per-round doit etre >= 0 (recu {proxy_per_round})")
    if not (0.0 <= distill_lambda <= 1.0):
        raise ValueError(
            f"fedavgkl-distill-lambda doit etre dans [0, 1] (recu {distill_lambda})")
    if distill_T <= 0.0:
        raise ValueError(
            f"fedavgkl-distill-temperature doit etre > 0 (recu {distill_T})")

    print(
        "[FedAvg+KL] INFO: agregation FedAvg des poids + distillation KL "
        "via proxy (regularisation secondaire)."
    )

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FedAvg+KL",
        strategy_class=FedAvgKLStrategy,
        strategy_kwargs={
            "proxy_size": proxy_size,
            "proxy_per_round": proxy_per_round,
            "distill_lambda": distill_lambda,
            "distill_temperature": distill_T,
            "num_classes": 10,  # CIFAR-10
        },
        train_config_fn=lambda r, lr, cfg: {
            "lr": lr,
            "round": r,
            "seed": int(cfg.get("seed", -1)),
        },
        project_dir_name="fedavg_kl",
        banner_extra=(
            f" proxy_size={proxy_size} proxy_per_round={proxy_per_round}"
            f" lambda={distill_lambda} T={distill_T}"
        ),
        extra_tail_fn=_fedavgkl_tail,
    )
