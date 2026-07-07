"""ServerApp FedProx (partitionnement IID ou non-IID configurable).

Identique a FedAvg cote serveur ; on broadcast 'mu' aux clients via train_config.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedProxStrategy

app = ServerApp()


def _fedprox_tail(_, strategy):
    """Diagnostic FedProx : moyenne ponderee de (mu/2) * ||w_local - w_global||^2.

    A comparer a la train_loss : prox << loss -> mu trop faible (drift non
    controle) ; prox >> loss -> mu trop grand (modele local fige).
    """
    prox = float(getattr(strategy, "last_prox_term_mean", 0.0))
    return f" prox={prox:.4f}"


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    # mu negatif ferait diverger l'optimisation prox (gradient repulsif)
    mu = float(cfg.get("mu", 0.01))
    if mu < 0.0:
        raise ValueError(f"FedProx exige mu >= 0 (recu mu={mu})")
    # L'agregation FedProx reste FedAvg ponderee par n_i, ce qui suppose le
    # meme E chez tous les clients.
    if int(cfg.get("epochs-heterogeneity", 0)):
        print("[fedprox] WARN: epochs-heterogeneity=1 active. FedProx "
              "suppose E identique entre clients (Li 2018) ; la ponderation "
              "par n_i devient biaisee sous E heterogene. Utilise FedNova "
              "si tu veux corriger tau_eff.")
    # au-dela de ~10, le terme proximal domine la CE loss et le modele local
    # ne bouge quasi plus. Valeurs typiques : 0.001-1.
    if mu > 10.0:
        print(f"[fedprox] WARN: mu={mu} tres eleve, le terme proximal va "
              "dominer la CE loss -> le modele local ne bougera quasi pas. "
              "Valeurs typiques : 0.001-1.")
    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FedProx",
        # FedProxStrategy capture le prox_term moyen pour diagnostic ;
        # l'aggregation reste FedAvg ponderee, conforme au papier.
        strategy_class=FedProxStrategy,
        strategy_kwargs={},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "mu": mu, "round": r},
        project_dir_name="fedprox",
        banner_extra=f" mu={mu}",
        extra_tail_fn=_fedprox_tail,
    )
