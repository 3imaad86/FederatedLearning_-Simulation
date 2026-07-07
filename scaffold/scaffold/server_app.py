"""ServerApp SCAFFOLD (Karimireddy et al. 2020).

Le serveur maintient :
  - w_global   : modele (gere par parent FedAvgStrategy)
  - c_global   : control variate global, init a zeros, recalcule chaque round

A chaque round :
  - Pack (w_global, c_global) dans un seul ArrayRecord (prefixes __cg__).
  - Aggrege les replies : nouveau w via FedAvg, c_new = c_global + mean(delta_c).
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import ScaffoldStrategy

app = ServerApp()


def _scaffold_tail(_, strategy):
    """Ajoute au log par round les normes de c_global et du delta_c agrege."""
    cg_norm = float(getattr(strategy, "last_c_global_norm", 0.0))
    dc_norm = float(getattr(strategy, "last_delta_c_norm", 0.0))
    return f" ||cG||={cg_norm:.2e} ||dC||={dc_norm:.2e}"


@app.main()
def main(grid: Grid, context: Context) -> None:
    momentum = float(context.run_config.get("momentum", 0.0))
    if momentum != 0.0:
        raise ValueError(
            "SCAFFOLD canonique exige momentum=0.0: le buffer momentum "
            "accumule la correction c_global-c_local et peut diverger. "
            f"Recu momentum={momentum}."
        )
    # La preuve de convergence de SCAFFOLD suppose le meme nombre d'epochs
    # chez tous les clients.
    if int(context.run_config.get("epochs-heterogeneity", 0)):
        print("[scaffold] WARN: epochs-heterogeneity=1 active. SCAFFOLD "
              "suppose K identique entre clients (Karimireddy 2020) ; les "
              "c_local n'ont pas la meme echelle sous E heterogene. La borne "
              "de convergence Thm 3 n'est plus garantie -- a documenter dans "
              "ton mémoire si tu compares.")
    # num-clients sert au facteur |S|/N de la mise a jour de c_global.
    n_total = int(context.run_config.get("num-clients", 10))
    server_lr = float(context.run_config.get("scaffold-server-lr", 1.0))
    weighted_aggregation = int(
        context.run_config.get("scaffold-weighted-aggregation", 1))
    if not (0.0 < server_lr <= 1.0):
        raise ValueError(
            f"SCAFFOLD exige 0 < scaffold-server-lr <= 1 "
            f"(recu {server_lr})"
        )
    if weighted_aggregation not in (0, 1):
        raise ValueError("scaffold-weighted-aggregation doit etre 0 ou 1")
    if int(context.run_config.get("data-heterogeneity", 0)) and not weighted_aggregation:
        print(
            "[scaffold] WARN: data-heterogeneity=1 avec aggregation uniforme. "
            "Les clients avec tres peu d'exemples pesent autant que les gros; "
            "pour comparer a FedAvg sample-weighted, utilise "
            "`scaffold-weighted-aggregation=1`."
        )
    mode_label = "canonical" if server_lr == 1.0 else "tuned"
    if server_lr != 1.0:
        print(
            f"[scaffold] INFO: server_lr={server_lr} (mode '{mode_label}'). "
            "Le papier Karimireddy 2020 (Algo 1) utilise eta_g=1.0 "
            "(mode 'canonical'). Pour reproduire strictement le papier, "
            "configure `scaffold-server-lr=1.0`."
        )
    # Liste des parametres du modele actif pour que c_global n'ait des
    # entrees que pour les vrais parametres (pas les buffers).
    from fl_common.data import get_model
    model_name = str(context.run_config.get("model-name", "net")).lower().strip()
    param_names = [name for name, _ in get_model(model_name).named_parameters()]

    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="SCAFFOLD",
        strategy_class=ScaffoldStrategy,
        strategy_kwargs={
            "num_clients_total": n_total,
            "server_lr": server_lr,
            "param_names": param_names,
            "weighted_aggregation": weighted_aggregation,
        },
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="scaffold",
        banner_extra=(
            f" server_lr={server_lr} ({mode_label})"
            f" weighted_agg={weighted_aggregation}"
        ),
        extra_tail_fn=_scaffold_tail,
    )
