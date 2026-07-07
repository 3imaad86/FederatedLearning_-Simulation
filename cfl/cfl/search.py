"""Submodel Search (Wang et al. 2023).

Pour chaque worker, on cherche le width qui maximise l'accuracy predite
par le predictor, sous contrainte de latence hardware. La recherche est
aleatoire sur `search_times` tirages.
"""

import random
from typing import Dict, List, Optional

from .predictor import AccuracyPredictorTrainer
from .submodel import (
    DEFAULT_CANDIDATE_WIDTHS,
    feasible_widths,
)


def select_submodel_for_worker(
    trainer: AccuracyPredictorTrainer,
    hardware_tier: int,
    search_times: int = 10,
    candidate_widths: Optional[List[float]] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Choisit le width d'un worker : recherche aleatoire parmi les widths
    respectant le budget latence, en maximisant l'accuracy predite."""
    if rng is None:
        rng = random
    cands = candidate_widths or DEFAULT_CANDIDATE_WIDTHS

    valid = feasible_widths(hardware_tier, cands)
    if not valid:
        return float(min(cands))

    if int(search_times) <= 0:
        return float(max(valid))

    best_w: Optional[float] = None
    best_acc = -1.0
    for _ in range(int(search_times)):
        w = rng.choice(valid)
        pred_acc = trainer.predict(w)
        if pred_acc > best_acc:
            best_acc = pred_acc
            best_w = w
    return float(best_w if best_w is not None else max(valid))


def select_submodels_for_all_workers(
    trainer: AccuracyPredictorTrainer,
    hardware_per_pid: Dict[int, int],
    num_clients: int,
    search_times: int = 10,
    candidate_widths: Optional[List[float]] = None,
    rng: Optional[random.Random] = None,
) -> List[float]:
    """Choisit un width par worker. Les pid jamais vus recoivent tier=1."""
    widths = []
    for pid in range(int(num_clients)):
        h = int(hardware_per_pid.get(pid, 1))
        w = select_submodel_for_worker(
            trainer, h,
            search_times=search_times,
            candidate_widths=candidate_widths,
            rng=rng,
        )
        widths.append(w)
    return widths
