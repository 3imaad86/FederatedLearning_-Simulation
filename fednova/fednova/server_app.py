"""ServerApp FedNova (Option C : rescaling 100% cote SERVEUR, fidele au papier).

Le client fait du SGD vanilla et envoie ses poids bruts + tau_i.
Le serveur calcule tau_eff a CHAQUE round (pas d'estimation a priori) :

    Delta_i = w_local_i - w_global
    tau_eff = sum (n_i/N) * tau_i              <- recalcule chaque round
    w_new   = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i)

Le `tau_eff` reel est affiche en fin de chaque round via `extra_tail_fn`.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedNovaStrategy

app = ServerApp()


def _fednova_tail(_, strategy):
    """Diagnostic FedNova : tau_eff + dispersion tau_i.

    tau_eff est la moyenne ponderee par n_i (facteur de rescaling utilise
    par le serveur). tau_min/tau_max sont les extremes parmi les clients.
    tau_ratio = max/min : si > 5, FedNova corrige beaucoup ; si ~1, FedNova
    degenere en FedAvg (peu de correction a apporter).
    """
    tau_eff = float(getattr(strategy, "last_tau_eff", 0.0))
    tau_min = float(getattr(strategy, "last_tau_min", 0.0))
    tau_max = float(getattr(strategy, "last_tau_max", 0.0))
    tau_ratio = float(getattr(strategy, "last_tau_ratio", 1.0))
    return (f" tau_eff={tau_eff:.1f} tau=[{tau_min:.0f},{tau_max:.0f}]"
            f" ratio={tau_ratio:.1f}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    momentum = float(context.run_config.get("momentum", 0.0))
    if momentum != 0.0:
        raise ValueError(
            "FedNova canonique exige momentum=0.0 cote client. "
            f"Recu momentum={momentum}. Utilise FedAvg pour tester SGD momentum, "
            "ou remets momentum=0.0 pour une comparaison FedNova valide."
        )
    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="FedNova",
        strategy_class=FedNovaStrategy,
        strategy_kwargs={},
        # Option C : aucune info FedNova-specifique a envoyer aux clients.
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fednova",
        extra_tail_fn=_fednova_tail,
    )
