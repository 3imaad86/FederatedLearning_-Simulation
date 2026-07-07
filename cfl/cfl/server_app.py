"""ServerApp CFL (Customized Federated Learning, Wang et al. 2023).

Le serveur :
  - heberge le parent Net (full width) via la strategy.
  - selectionne le submodel de chaque worker (Algo 1) avant chaque round.
  - agrege les replies via width expansion (Algo 3) en moyenne canal-par-canal.
  - entraine le predictor (Algo 2) avec les samples (q_k, w_k, acc_k) collectes.

Hyperparametres CFL exposes via pyproject.toml :
  - cfl-search-times      : S iterations de l'Algo 1 (defaut 10)
  - cfl-predictor-lr      : LR Adam pour le predictor (defaut 0.01)
  - cfl-predictor-hidden  : taille des couches cachees (defaut 32)
  - cfl-candidate-widths  : liste des ratios candidats (defaut [0.25, 0.5, 0.75, 1.0])
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training

from .strategy import CFLStrategy

app = ServerApp()


def _cfl_tail(_, strategy):
    """Diagnostic CFL : distribution des widths + sante du predictor."""
    dist = getattr(strategy, "last_width_dist", {}) or {}
    # Format compact : w=0.25:N1,0.5:N2,...
    widths_str = ",".join(
        f"{w:.2f}:{n}" for w, n in sorted(dist.items())
    )
    # Predictor : taille buffer + derniere loss
    trainer = getattr(strategy, "predictor_trainer", None)
    pred_n = int(getattr(trainer, "last_n_samples", 0))
    pred_loss = float(getattr(trainer, "last_loss", 0.0))
    return f" widths=[{widths_str}] pred_n={pred_n} pred_loss={pred_loss:.4f}"


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    num_clients = int(cfg.get("num-clients", 10))
    search_times = int(cfg.get("cfl-search-times", 10))
    predictor_lr = float(cfg.get("cfl-predictor-lr", 0.01))
    predictor_hidden = int(cfg.get("cfl-predictor-hidden", 32))
    # cfl-candidate-widths peut etre une string TOML "0.25,0.5,0.75,1.0"
    # ou une liste/tuple. On normalise en liste de float.
    raw_widths = cfg.get("cfl-candidate-widths", "0.25,0.5,0.75,1.0")
    if isinstance(raw_widths, str):
        candidate_widths = [float(x.strip()) for x in raw_widths.split(",")
                            if x.strip()]
    else:
        candidate_widths = [float(x) for x in raw_widths]
    seed = int(cfg.get("seed", -1))

    # Validations CFL
    if search_times < 1:
        raise ValueError("cfl-search-times doit etre >= 1")
    if predictor_lr <= 0.0:
        raise ValueError("cfl-predictor-lr doit etre > 0")
    if predictor_hidden < 4:
        raise ValueError("cfl-predictor-hidden doit etre >= 4")
    if not candidate_widths:
        raise ValueError("cfl-candidate-widths doit etre non vide")
    if any(w <= 0.0 or w > 1.0 for w in candidate_widths):
        raise ValueError("cfl-candidate-widths doit avoir des valeurs dans ]0, 1]")

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="CFL",
        strategy_class=CFLStrategy,
        strategy_kwargs={
            "num_clients": num_clients,
            "search_times": search_times,
            "predictor_lr": predictor_lr,
            "predictor_hidden": predictor_hidden,
            "candidate_widths": candidate_widths,
            "seed": seed,
        },
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="cfl",
        banner_extra=(
            f" widths={candidate_widths} S={search_times}"
            f" pred_lr={predictor_lr}"
        ),
        extra_tail_fn=_cfl_tail,
    )
