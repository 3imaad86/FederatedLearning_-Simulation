"""ServerApp FedAvg (partitionnement IID ou non-IID configurable)."""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedAvgStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    # FedAvg suppose que tous les clients font le meme nombre d'epochs.
    # Avec epochs-heterogeneity=1, ce n'est plus le cas : prefere FedNova
    # ou SCAFFOLD pour une comparaison theorique propre.
    if int(cfg.get("epochs-heterogeneity", 0)):
        print("[fedavg] WARN: epochs-heterogeneity=1 active. FedAvg suppose "
              "E identique chez tous les clients (McMahan 2017). La "
              "ponderation par n_i devient biaisee -- utilise FedNova si "
              "tu veux corriger tau_eff.")
    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FedAvg",
        strategy_class=FedAvgStrategy,
        strategy_kwargs={},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fedavg",
    )
