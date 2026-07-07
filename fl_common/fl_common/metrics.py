"""Metriques FL (efficacite + fairness). Produit les CSV dans results/.

Eval 100% cote SERVEUR sur le test set CIFAR-10 (10k IID).
Toutes les metriques par-classe sont donc non biaisees par le non-IID.
"""

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


NUM_CLASSES = 10

RESULTS_DIR = os.environ.get(
    "FL_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "results"),
)
GLOBAL_CSV = os.path.join(RESULTS_DIR, "metrics_global.csv")
PER_CLASS_CSV = os.path.join(RESULTS_DIR, "metrics_per_class.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
PARTICIPATION_CSV = os.path.join(RESULTS_DIR, "metrics_participation.csv")
CLIENTS_CSV = os.path.join(RESULTS_DIR, "metrics_clients.csv")

GLOBAL_HEADER = [
    "round", "accuracy", "loss", "macro_recall", "macro_f1",
    "jfi_classes", "worst_class_acc", "acc_variance_classes", "min_max_class_gap",
    # comm_cost_mb = bytes WAN, comm_lan_mb = bytes LAN (HFL uniquement),
    # technical_comm_mb = payload Flower reellement transporte.
    "comm_cost_mb",
    "comm_lan_mb",
    "technical_comm_mb",
    "round_time_s", "mean_client_time_s", "max_client_time_s",
    "mean_comm_time_s", "max_comm_time_s",
    "edge_cloud_time_s", "edge_cloud_downlink_time_s", "edge_cloud_uplink_time_s",
    "mean_epochs_used", "mean_resource_tier",
    "energy_j_round", "energy_j_cumulative",
    "client_energy_j_round", "edge_cloud_energy_j_round",
    "edge_cloud_energy_j_cumulative",
    "compute_energy_j_round", "comm_energy_j_round",
    "compute_energy_j_cumulative", "comm_energy_j_cumulative",
    "local_loss", "local_acc",
    "fedstrag_buffered", "fedstrag_applied", "fedstrag_pending",
    "fedeve_kalman_gain", "fedeve_sigma_Q", "fedeve_sigma_R",
    # fairness inter-clients sur local_acc
    "jfi_clients_acc", "client_acc_variance", "client_acc_gap",
    "worst_client_acc", "best_client_acc",
    # diagnostics algo-specifiques (0.0 si non applicable)
    "cfl_mean_width", "cfl_predictor_loss", "cfl_n_active_widths",
    "scaffold_cg_norm", "scaffold_dc_norm",
    "fednova_tau_eff", "fednova_tau_min", "fednova_tau_max",
    "fedprox_prox_term",
    "hfl_battery_min_ratio", "hfl_battery_mean_ratio",
    "hfl_queue_mean", "hfl_queue_max",
    # breakdown energie + width par tier hardware
    "tier0_energy_j_round", "tier1_energy_j_round", "tier2_energy_j_round",
    "tier0_n_clients", "tier1_n_clients", "tier2_n_clients",
    "tier0_mean_width", "tier1_mean_width", "tier2_mean_width",
]
PER_CLASS_HEADER = ["round"] + [f"class_{i}" for i in range(NUM_CLASSES)]
SUMMARY_HEADER = [
    "total_time_s", "rtc90", "rounds_to_50", "rounds_to_70", "rounds_to_90",
    "participation_jfi", "worst_participation", "best_participation",
    # Ressources cumulees pour atteindre acc=0.3/0.5/0.7
    "energy_to_30_J", "energy_to_50_J", "energy_to_70_J",
    "comm_to_30_MB", "comm_to_50_MB", "comm_to_70_MB",
    "time_to_30_s", "time_to_50_s", "time_to_70_s",
]
PARTICIPATION_HEADER = ["client_id", "times_selected"]
CLIENTS_HEADER = [
    "round", "client_id", "num_examples", "epochs_used",
    "resource_tier", "net_tier", "is_lan", "hfl_edge_id",
    "local_time_s", "local_eval_time_s", "comm_time_s",
    "energy_j", "compute_energy_j", "comm_energy_j",
    "dropped", "battery_remaining_j", "battery_capacity_j",
    "battery_constrained", "local_loss", "local_acc",
    "fedstrag_late", "fedstrag_staleness", "deadline_miss_s",
    "server_wait_time_s",
]


def set_results_dir(path):
    """Configure le dossier de resultats pour le run courant.

    Invalide aussi le cache du test-loader de server_evaluate (utile si
    plusieurs runs se suivent dans le meme process).
    """
    global RESULTS_DIR, GLOBAL_CSV, PER_CLASS_CSV, SUMMARY_CSV
    global PARTICIPATION_CSV, CLIENTS_CSV
    RESULTS_DIR = str(Path(path).resolve())
    GLOBAL_CSV = os.path.join(RESULTS_DIR, "metrics_global.csv")
    PER_CLASS_CSV = os.path.join(RESULTS_DIR, "metrics_per_class.csv")
    SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    PARTICIPATION_CSV = os.path.join(RESULTS_DIR, "metrics_participation.csv")
    CLIENTS_CSV = os.path.join(RESULTS_DIR, "metrics_clients.csv")
    # Invalide le cache test-loader (import local pour eviter le cycle).
    try:
        from .server_runner import server_evaluate as _server_evaluate
        _server_evaluate._cache = {}
    except (ImportError, AttributeError):
        # Cache pas encore initialise -> rien a faire.
        pass


def get_results_dir():
    """Retourne le dossier de resultats actif."""
    return RESULTS_DIR


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def reset_files():
    ensure_dir()
    for p in (GLOBAL_CSV, PER_CLASS_CSV, SUMMARY_CSV,
              PARTICIPATION_CSV, CLIENTS_CSV):
        if os.path.exists(p):
            os.remove(p)


def _append(path, header, row):
    """Append une ligne (cree le fichier + header si absent)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _overwrite(path, header, rows):
    """Reecrit le fichier complet (header + rows)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def resolve_dst_results_dir(project_dir_name):
    """Cherche le dossier `<project>/results` dans cwd ou parents.

    Flower execute le code depuis un cache temporaire, donc on copie les
    CSV dans le vrai dossier projet pour qu'ils soient persistants.
    """
    if "FL_RESULTS_DIR" in os.environ:
        return os.environ["FL_RESULTS_DIR"]

    def is_project(p):
        return p.is_dir() and p.name == project_dir_name and (p / "pyproject.toml").exists()

    cwd = Path(os.getcwd()).resolve()
    candidates = [cwd] + list(cwd.parents)
    for parent in candidates:
        if is_project(parent):
            return str(parent / "results")
        try:
            for child in parent.iterdir():
                if is_project(child):
                    return str(child / "results")
        except OSError:
            continue
    return str(cwd / "results")


def jains_fairness_index(values):
    """JFI = (sum xi)^2 / (n * sum xi^2). 1 = parfaitement equitable."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or (arr * arr).sum() == 0.0:
        return 0.0
    return float(arr.sum() ** 2 / (arr.size * (arr * arr).sum()))


def class_accuracies_from_preds(y_true, y_pred, num_classes=NUM_CLASSES):
    """Accuracy par classe a partir d'une matrice de confusion."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return [0.0] * num_classes
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return [float(cm[c, c] / cm[c].sum()) if cm[c].sum() > 0 else 0.0
            for c in range(num_classes)]


def macro_recall_f1_from_preds(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0
    return (
        float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def rounds_to_convergence(accuracies, ratio=0.9):
    """Nb de rounds pour atteindre `ratio` × max(accuracies)."""
    accs = list(accuracies)
    if not accs:
        return None
    threshold = ratio * max(accs)
    for i, a in enumerate(accs, start=1):
        if a >= threshold:
            return i
    return None


def rounds_to_target(accuracies, target):
    """Nb de rounds pour atteindre `target` (ex: 0.5)."""
    for i, a in enumerate(accuracies, start=1):
        if a >= target:
            return i
    return None


def _normalize_class_accs(class_accuracies):
    """Pad/tronque la liste pour avoir exactement NUM_CLASSES."""
    accs = [float(a) for a in (class_accuracies or [])]
    if len(accs) < NUM_CLASSES:
        return accs + [0.0] * (NUM_CLASSES - len(accs))
    return accs[:NUM_CLASSES]


def log_round(server_round, accuracy, loss, macro_recall, macro_f1,
              class_accuracies,
              comm_cost_mb=0.0, comm_lan_mb=0.0,
              technical_comm_mb=0.0,
              round_time_s=0.0, mean_client_time_s=0.0, max_client_time_s=0.0,
              mean_comm_time_s=0.0, max_comm_time_s=0.0,
              edge_cloud_time_s=0.0,
              edge_cloud_downlink_time_s=0.0,
              edge_cloud_uplink_time_s=0.0,
              mean_epochs_used=0.0, mean_resource_tier=0.0,
              energy_j_round=0.0, energy_j_cumulative=0.0,
              client_energy_j_round=0.0,
              edge_cloud_energy_j_round=0.0,
              edge_cloud_energy_j_cumulative=0.0,
              compute_energy_j_round=0.0, comm_energy_j_round=0.0,
              compute_energy_j_cumulative=0.0, comm_energy_j_cumulative=0.0,
              local_loss=0.0, local_acc=0.0,
              fedstrag_buffered=0, fedstrag_applied=0, fedstrag_pending=0,
              fedeve_kalman_gain=0.0, fedeve_sigma_Q=0.0, fedeve_sigma_R=0.0,
              # fairness inter-clients
              jfi_clients_acc=0.0, client_acc_variance=0.0,
              client_acc_gap=0.0, worst_client_acc=0.0, best_client_acc=0.0,
              # diagnostics algo-specifiques (0.0 si absent)
              cfl_mean_width=0.0, cfl_predictor_loss=0.0,
              cfl_n_active_widths=0,
              scaffold_cg_norm=0.0, scaffold_dc_norm=0.0,
              fednova_tau_eff=0.0, fednova_tau_min=0.0, fednova_tau_max=0.0,
              fedprox_prox_term=0.0,
              hfl_battery_min_ratio=0.0, hfl_battery_mean_ratio=0.0,
              hfl_queue_mean=0.0, hfl_queue_max=0.0,
              # breakdown par tier
              tier0_energy_j_round=0.0, tier1_energy_j_round=0.0,
              tier2_energy_j_round=0.0,
              tier0_n_clients=0, tier1_n_clients=0, tier2_n_clients=0,
              tier0_mean_width=0.0, tier1_mean_width=0.0,
              tier2_mean_width=0.0):
    """Ecrit une ligne dans metrics_global.csv + metrics_per_class.csv.

    Les kwargs de diagnostics algo-specifiques sont optionnels (defaut 0.0).
    """
    ensure_dir()
    class_accs = _normalize_class_accs(class_accuracies)

    _append(GLOBAL_CSV, GLOBAL_HEADER, [
        server_round, float(accuracy), float(loss),
        float(macro_recall), float(macro_f1),
        jains_fairness_index(class_accs),
        float(min(class_accs)),
        float(np.var(class_accs)),
        float(max(class_accs) - min(class_accs)),
        float(comm_cost_mb), float(comm_lan_mb), float(technical_comm_mb),
        float(round_time_s), float(mean_client_time_s), float(max_client_time_s),
        float(mean_comm_time_s), float(max_comm_time_s),
        float(edge_cloud_time_s), float(edge_cloud_downlink_time_s),
        float(edge_cloud_uplink_time_s),
        float(mean_epochs_used), float(mean_resource_tier),
        float(energy_j_round), float(energy_j_cumulative),
        float(client_energy_j_round), float(edge_cloud_energy_j_round),
        float(edge_cloud_energy_j_cumulative),
        float(compute_energy_j_round), float(comm_energy_j_round),
        float(compute_energy_j_cumulative), float(comm_energy_j_cumulative),
        float(local_loss), float(local_acc),
        int(fedstrag_buffered), int(fedstrag_applied), int(fedstrag_pending),
        float(fedeve_kalman_gain), float(fedeve_sigma_Q), float(fedeve_sigma_R),
        float(jfi_clients_acc), float(client_acc_variance),
        float(client_acc_gap), float(worst_client_acc), float(best_client_acc),
        float(cfl_mean_width), float(cfl_predictor_loss),
        int(cfl_n_active_widths),
        float(scaffold_cg_norm), float(scaffold_dc_norm),
        float(fednova_tau_eff), float(fednova_tau_min), float(fednova_tau_max),
        float(fedprox_prox_term),
        float(hfl_battery_min_ratio), float(hfl_battery_mean_ratio),
        float(hfl_queue_mean), float(hfl_queue_max),
        # breakdown par tier
        float(tier0_energy_j_round), float(tier1_energy_j_round),
        float(tier2_energy_j_round),
        int(tier0_n_clients), int(tier1_n_clients), int(tier2_n_clients),
        float(tier0_mean_width), float(tier1_mean_width),
        float(tier2_mean_width),
    ])
    _append(PER_CLASS_CSV, PER_CLASS_HEADER, [server_round] + class_accs)


def log_client_details(server_round, details):
    """Ecrit une ligne par client dans metrics_clients.csv."""
    ensure_dir()
    for d in sorted(details or [], key=lambda row: row.get("pid", -1)):
        _append(CLIENTS_CSV, CLIENTS_HEADER, [
            int(server_round),
            int(d.get("pid", -1)),
            int(d.get("n", 0)),
            float(d.get("epochs", 0.0)),
            float(d.get("tier", 0.0)),
            int(d.get("net_tier", -1)),
            int(d.get("is_lan", 0)),
            int(d.get("hfl_edge_id", -1)),
            float(d.get("time", 0.0)),
            float(d.get("local_eval_time", 0.0)),
            float(d.get("comm_time", 0.0)),
            float(d.get("energy", 0.0)),
            float(d.get("energy_compute", 0.0)),
            float(d.get("energy_comm", 0.0)),
            int(d.get("dropped", 0)),
            float(d.get("battery_remaining_j", -1.0)),
            float(d.get("battery_capacity_j", -1.0)),
            int(d.get("battery_constrained", -1)),
            float(d.get("local_loss", 0.0)),
            float(d.get("local_acc", 0.0)),
            int(d.get("fedstrag_late", 0)),
            float(d.get("fedstrag_staleness", 0.0)),
            float(d.get("deadline_miss_s", 0.0)),
            float(d.get("server_wait_time", 0.0)),
        ])


def _participation_counts_list(counts, num_clients=None):
    """Convertit le dict {pid: count} -> liste alignee sur range(num_clients)."""
    if isinstance(counts, dict):
        if num_clients is not None:
            return [int(counts.get(cid, 0)) for cid in range(int(num_clients))]
        return list(counts.values())
    return list(counts) if counts else []


def resource_to_target(accs_history, resource_history, target):
    """Ressource cumulee (energie, comm, temps) pour atteindre un seuil
    d'accuracy. `resource_history` est par round (cumule en interne).
    Retourne None si le seuil n'est jamais atteint."""
    if not accs_history or not resource_history:
        return None
    cumul = 0.0
    n = min(len(accs_history), len(resource_history))
    for i in range(n):
        cumul += float(resource_history[i])
        if accs_history[i] >= target:
            return cumul
    return None


def log_summary(total_time_s, accs_history, participation_counts,
                targets=(0.5, 0.7, 0.9), num_clients=None,
                energy_per_round=None, comm_per_round=None, time_per_round=None):
    """Ecrit metrics_summary.csv (rtc90, rounds-to-targets, participation fairness).

    Si `energy_per_round`, `comm_per_round`, `time_per_round` sont fournis
    (listes alignees avec `accs_history`), on calcule aussi les
    ressources-cumulees-to-target pour les seuils 0.3/0.5/0.7.
    """
    ensure_dir()
    rtc90 = rounds_to_convergence(accs_history, ratio=0.9)
    rt = [rounds_to_target(accs_history, t) for t in targets]
    counts = _participation_counts_list(participation_counts, num_clients)

    # Pareto-efficacite
    pareto_targets = (0.3, 0.5, 0.7)
    energy_to = [None] * 3
    comm_to = [None] * 3
    time_to = [None] * 3
    if energy_per_round:
        energy_to = [resource_to_target(accs_history, energy_per_round, t)
                     for t in pareto_targets]
    if comm_per_round:
        comm_to = [resource_to_target(accs_history, comm_per_round, t)
                   for t in pareto_targets]
    if time_per_round:
        time_to = [resource_to_target(accs_history, time_per_round, t)
                   for t in pareto_targets]

    def _fmt(v):
        return float(v) if v is not None else ""

    _overwrite(SUMMARY_CSV, SUMMARY_HEADER, [[
        float(total_time_s),
        rtc90 if rtc90 is not None else "",
        rt[0] if rt[0] is not None else "",
        rt[1] if rt[1] is not None else "",
        rt[2] if rt[2] is not None else "",
        jains_fairness_index(counts) if counts else 0.0,
        int(min(counts)) if counts else 0,
        int(max(counts)) if counts else 0,
        _fmt(energy_to[0]), _fmt(energy_to[1]), _fmt(energy_to[2]),
        _fmt(comm_to[0]), _fmt(comm_to[1]), _fmt(comm_to[2]),
        _fmt(time_to[0]), _fmt(time_to[1]), _fmt(time_to[2]),
    ]])


def log_participation(participation_counts, num_clients=None):
    """Ecrit metrics_participation.csv (combien de fois chaque client a participe)."""
    ensure_dir()
    if num_clients is not None:
        rows = [[cid, int(participation_counts.get(cid, 0))]
                for cid in range(int(num_clients))]
    else:
        rows = [[int(cid), int(n)]
                for cid, n in sorted(participation_counts.items())]
    _overwrite(PARTICIPATION_CSV, PARTICIPATION_HEADER, rows)
