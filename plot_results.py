"""Plots de comparaison FedAvg / FedProx / FedNova / SCAFFOLD / FedEve / CFL / FedAvg+KL / q-FedAvg / FairFed / HFL-BEFL.

Lit les CSV dans chaque `<algo>/results/` et produit des PNG dans `plots/`.

Usage :
    python plot_results.py                          # tous les algos trouves
    python plot_results.py fedavg hfl_befl          # uniquement ceux listes
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ALGOS = ["fedavg", "fedprox", "fednova", "scaffold", "fedeve", "cfl",
         "fedavg_kl", "qfedavg", "fairfed",
         "hfl_befl", "hfl_befl_cfl"]
COLORS = {"fedavg": "tab:blue", "fedprox": "tab:orange",
          "fednova": "tab:green", "scaffold": "tab:purple",
          "fedeve": "tab:olive", "cfl": "tab:red",
          "fedavg_kl": "tab:brown",
          "qfedavg": "tab:gray",
          "fairfed": "magenta",
          "hfl_befl": "tab:cyan", "hfl_befl_cfl": "tab:pink"}

# Libelles affiches dans les plots (par defaut : nom du dossier en majuscules).
LABELS = {"fedavg_kl": "FEDAVG+KL"}


def algo_label(algo):
    return LABELS.get(algo, algo.upper())


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def _safe_read_csv(path):
    """Lit un CSV en tolerant fichier absent / vide / corrompu.

    Retourne un DataFrame vide si le CSV est inutilisable, ce qui permet aux
    plots de continuer (les fonctions plot_* skippent silencieusement les
    DataFrames vides ou les colonnes manquantes).
    """
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        print(f"[warn] CSV illisible : {path} ({exc})")
        return None


def load_csvs(algo):
    """Retourne (global_df, per_class_df, summary_df) ou None si global absent."""
    results = os.path.join(ROOT, algo, "results")
    gdf = _safe_read_csv(os.path.join(results, "metrics_global.csv"))
    if gdf is None or gdf.empty:
        return None
    # per_class et summary peuvent etre absents -> DataFrame vide.
    pdf = _safe_read_csv(os.path.join(results, "metrics_per_class.csv"))
    if pdf is None:
        pdf = pd.DataFrame()
    sdf = _safe_read_csv(os.path.join(results, "metrics_summary.csv"))
    if sdf is None:
        sdf = pd.DataFrame()
    return gdf, pdf, sdf


def load_client_csv(algo):
    """Retourne metrics_clients.csv si disponible."""
    return _safe_read_csv(
        os.path.join(ROOT, algo, "results", "metrics_clients.csv"))


def plot_curve(data, y_col, ylabel, title, filename, cumulative=False):
    """Plot une courbe par algo pour la colonne `y_col`."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, (gdf, _, _) in data.items():
        if y_col not in gdf.columns:
            continue
        y = gdf[y_col].cumsum() if cumulative else gdf[y_col]
        ax.plot(gdf["round"], y, marker="o", label=algo_label(algo),
                color=COLORS[algo], linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"  -> {filename}")


def plot_curve_fallback(data, y_cols, ylabel, title, filename, cumulative=False):
    """Plot avec choix de colonne par priorite, utile pour CSV anciens."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for algo, (gdf, _, _) in data.items():
        y_col = next((col for col in y_cols if col in gdf.columns), None)
        if y_col is None:
            continue
        y = gdf[y_col].cumsum() if cumulative else gdf[y_col]
        ax.plot(gdf["round"], y, marker="o", label=algo_label(algo),
                color=COLORS[algo], linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"  -> {filename}")


def plot_server_vs_local(data, server_col, local_col, ylabel, title, filename):
    """Overlay server (ligne pleine) vs local agrege (ligne pointillee) par algo.

    Le gap entre les 2 courbes visualise le biais non-IID : chaque client
    overfitte sa propre distribution -> sa metrique locale est sur-estimee
    par rapport a l'eval serveur sur un test set IID.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, (gdf, _, _) in data.items():
        color = COLORS[algo]
        ax.plot(gdf["round"], gdf[server_col], marker="o", linestyle="-",
                color=color, linewidth=2, label=f"{algo_label(algo)} (server)")
        if local_col in gdf.columns:
            ax.plot(gdf["round"], gdf[local_col], marker="x", linestyle="--",
                    color=color, linewidth=1.5, alpha=0.8,
                    label=f"{algo_label(algo)} (local agg)")
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"  -> {filename}")


def plot_per_class_heatmap(data):
    """Grille de heatmaps per-class accuracy par algo.

    Tolerant aux algos sans per_class CSV (skip silencieusement). Le nombre
    de classes est calcule dynamiquement depuis les colonnes (au lieu du 10
    hardcode pour CIFAR-10).
    """
    # Filtre les algos qui ont un per_class non vide
    valid = {a: (_, pdf, _) for a, (_, pdf, _) in data.items()
             if pdf is not None and not pdf.empty
             and any(c.startswith("class_") for c in pdf.columns)}
    if not valid:
        return
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for i, (algo, (_, pdf, _)) in enumerate(valid.items()):
        ax = axes[0, i]
        class_cols = [c for c in pdf.columns if c.startswith("class_")]
        n_classes = len(class_cols)
        mat = pdf[class_cols].values.T  # classes en lignes, rounds en colonnes
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0,
                       extent=[0.5, len(pdf) + 0.5,
                               n_classes - 0.5, -0.5])
        ax.set_title(algo_label(algo))
        ax.set_xlabel("Round")
        ax.set_ylabel("Classe")
        ax.set_yticks(range(n_classes))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Accuracy par classe (eval serveur)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "per_class_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  -> per_class_heatmap.png")


def plot_summary_bars(data):
    """Bar chart des rounds-to-target (rtc90, r50, r70, r90) par algo.

    Utilise NaN pour les cibles non atteintes (pas 0, qui serait
    indistinguable d'une vraie valeur zero) : matplotlib bar() affiche un
    trou visible pour NaN, donc l'absence de donnee reste lisible.
    """
    import numpy as np
    metrics = ["rtc90", "rounds_to_50", "rounds_to_70", "rounds_to_90"]
    labels = ["RTC90", "R->0.50", "R->0.70", "R->0.90"]
    # Filtre les algos sans summary (sdf vide tolere)
    valid = {a: (_, _, s) for a, (_, _, s) in data.items()
             if s is not None and not s.empty}
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(metrics))
    width = 0.8 / max(len(valid), 1)
    for i, (algo, (_, _, sdf)) in enumerate(valid.items()):
        vals = []
        for m in metrics:
            v = sdf[m].iloc[0] if m in sdf.columns else None
            vals.append(float(v) if pd.notna(v) else np.nan)
        ax.bar([xi + i * width for xi in x], vals, width=width,
               label=algo_label(algo), color=COLORS[algo])
    ax.set_xticks([xi + width * (len(valid) - 1) / 2 for xi in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Nombre de rounds (vide = cible non atteinte)")
    ax.set_title("Vitesse de convergence par algorithme")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_bars.png"), dpi=120)
    plt.close(fig)
    print(f"  -> summary_bars.png")


def plot_client_energy(client_data):
    """Graphe la consommation energetique de chaque client par algo."""
    for algo, cdf in client_data.items():
        needed = {"round", "client_id", "energy_j"}
        if cdf is None or not needed.issubset(set(cdf.columns)):
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.get_cmap("tab20")
        for i, (cid, rows) in enumerate(cdf.groupby("client_id")):
            rows = rows.sort_values("round")
            ax.plot(rows["round"], rows["energy_j"].cumsum(),
                    marker="o", linewidth=1.8, markersize=4,
                    color=cmap(i % 20), label=f"client {int(cid)}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Energie client cumulee (J)")
        ax.set_title(f"Consommation cumulee par client - {algo_label(algo)}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        filename = f"energy_clients_cumulative_{algo}.png"
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
        plt.close(fig)
        print(f"  -> {filename}")

        totals = (cdf.groupby("client_id")["energy_j"]
                    .sum()
                    .sort_index())
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([str(int(cid)) for cid in totals.index], totals.values,
               color=[cmap(i % 20) for i in range(len(totals))])
        ax.set_xlabel("Client")
        ax.set_ylabel("Energie totale client (J)")
        ax.set_title(f"Consommation totale par client - {algo_label(algo)}")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        filename = f"energy_clients_total_{algo}.png"
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
        plt.close(fig)
        print(f"  -> {filename}")


def main():
    algos = sys.argv[1:] if len(sys.argv) > 1 else ALGOS
    data = {}
    client_data = {}
    for algo in algos:
        res = load_csvs(algo)
        if res is None:
            print(f"[skip] {algo} : metrics_global.csv absent")
            continue
        data[algo] = res
        cdf = load_client_csv(algo)
        if cdf is not None:
            client_data[algo] = cdf
        print(f"[ok]   {algo} : {len(res[0])} rounds")

    if not data:
        print("Aucun resultat a tracer. Lance d'abord `flwr run .` dans un dossier d'algo.")
        return

    print(f"\nSauvegarde dans {OUT_DIR}:")
    plot_curve(data, "accuracy", "Accuracy",
               "Accuracy (eval serveur CIFAR-10 IID)", "accuracy_curve.png")
    plot_curve(data, "loss", "Loss",
               "Loss (eval serveur)", "loss_curve.png")
    plot_curve(data, "macro_f1", "Macro F1",
               "Macro F1", "macro_f1_curve.png")
    plot_curve(data, "jfi_classes", "Jain's Fairness Index (classes)",
               "Fairness inter-classes", "fairness_curve.png")
    plot_curve(data, "min_max_class_gap", "max(class_acc) - min(class_acc)",
               "Ecart entre meilleure et pire classe", "class_gap_curve.png")
    plot_curve(data, "comm_cost_mb", "Comm cumule (MB)",
               "Cout de communication cumule", "comm_cost_curve.png",
               cumulative=True)
    plot_curve(data, "round_time_s", "Temps simule round (s)",
               "Temps simule par round", "round_time_curve.png")
    # Energie CLIENTS uniquement. `energy_j_round` inclut aussi edge-cloud
    # dans les CSV recents HFL; on prefere donc `client_energy_j_round`.
    plot_curve_fallback(data, ["client_energy_j_round", "energy_j_round"],
                        "Energie clients (J) / round",
                        "Energie clients simulee par round",
                        "energy_round_curve.png")
    plot_curve_fallback(data, ["client_energy_j_round", "energy_j_round"],
                        "Energie clients cumulee (J)",
                        "Energie clients simulee cumulee",
                        "energy_cumulative_curve.png",
                        cumulative=True)
    plot_curve(data, "edge_cloud_energy_j_round", "Energie edge-cloud (J) / round",
               "Energie edge-cloud par round", "energy_edge_cloud_curve.png")

    # Overlays server vs local agrege (visualise le biais non-IID)
    plot_server_vs_local(data, "accuracy", "local_acc", "Accuracy",
                         "Server (IID) vs local agg (non-IID bias)",
                         "accuracy_server_vs_local.png")
    plot_server_vs_local(data, "loss", "local_loss", "Loss",
                         "Server (IID) vs local agg (non-IID bias)",
                         "loss_server_vs_local.png")

    plot_per_class_heatmap(data)
    plot_summary_bars(data)
    if client_data:
        plot_client_energy(client_data)
    else:
        print("  -> metrics_clients.csv absent: relance les algos pour tracer par client")
    print(f"\n{len(data)} algo(s) tracees.")


if __name__ == "__main__":
    main()
