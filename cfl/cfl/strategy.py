"""CFLStrategy : Customized Federated Learning (Wang et al. 2023).

  - configure_train : selectionne un width par client (predictor + search)
    et l'injecte dans `cfl-widths-per-pid`.
  - aggregate_train : moyenne canal-par-canal (width expansion), un canal
    n'est moyenne que parmi les clients qui l'avaient actif.
"""

import random
from typing import Dict, List

import torch
from flwr.app import ArrayRecord

from fl_common.strategy import FedAvgStrategy, _is_dropped, _no_model_transfer

from .predictor import AccuracyPredictor, AccuracyPredictorTrainer
from .search import select_submodels_for_all_workers
from .submodel import (
    DEFAULT_CANDIDATE_WIDTHS,
    active_channel_mask,
    submodel_to_parent_zero_pad,
)


class CFLStrategy(FedAvgStrategy):
    """CFL : submodel sampling + width-aware aggregation + predictor."""

    def __init__(
        self,
        *args,
        num_clients: int = 10,
        search_times: int = 10,
        predictor_lr: float = 0.01,
        predictor_hidden: int = 32,
        candidate_widths: List[float] = None,
        seed: int = -1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_clients = int(num_clients)
        self.search_times = int(search_times)
        self.candidate_widths = list(candidate_widths or DEFAULT_CANDIDATE_WIDTHS)

        # predictor + trainer
        self.predictor = AccuracyPredictor(hidden_dim=int(predictor_hidden))
        self.predictor_trainer = AccuracyPredictorTrainer(
            self.predictor, lr=float(predictor_lr))

        # tier hardware par pid, alimente au fil des replies
        self.hardware_per_pid: Dict[int, int] = {}

        # RNG dedie a la recherche, reseed a chaque round si seed >= 0.
        self.seed = int(seed)
        self._search_rng_seed_base = (
            self.seed * 6829 + 13 if self.seed >= 0 else None)
        self._search_rng = random.Random(self._search_rng_seed_base)

        # diagnostics
        self.last_width_dist: Dict[float, int] = {}
        self.last_widths: List[float] = []
        # pid effectivement samples ce round (pour _mean_width_sq sous
        # fraction_train < 1.0)
        self.last_sampled_pids: List[int] = []

    def _mean_width_sq(self, contents=None) -> float:
        """Moyenne des width^2 des clients du round (ratio de bytes CFL).

        Sous fraction_train < 1.0, le fallback filtre last_widths aux pid
        reellement samples pour ne pas compter des intentions non realisees.
        """
        widths = []
        for content in contents or []:
            try:
                if _no_model_transfer(content):
                    continue
                mr = next(iter(content.metric_records.values()))
                widths.append(float(mr.get("cfl_width_ratio", 1.0)))
            except (AttributeError, ValueError, StopIteration):
                continue
        if not widths:
            if self.last_sampled_pids and self.last_widths:
                widths = [
                    float(self.last_widths[pid])
                    for pid in self.last_sampled_pids
                    if 0 <= pid < len(self.last_widths)
                ]
            else:
                widths = [float(w) for w in (self.last_widths or [])]
        if not widths:
            return 1.0
        return sum(w * w for w in widths) / len(widths)

    def _apply_cfl_downlink_accounting(self, download_count: int, contents=None):
        """Expose le cout logique submodel et le payload technique full-parent
        (Flower transporte le parent complet, le client extrait son submodel)."""
        technical = int(
            self._last_downlink_record_bytes
            * int(download_count)
            * self.comm_size_ratio
        )
        logical = int(technical * self._mean_width_sq(contents))
        self.last_downlink_bytes = logical
        self.last_technical_downlink_bytes = technical

    def _refresh_direct_downlink_bytes(self, contents):
        super()._refresh_direct_downlink_bytes(contents)
        self._apply_cfl_downlink_accounting(self._download_count(contents), contents)

    def configure_train(self, server_round, arrays, config, grid):
        """Selectionne le width de chaque worker via le predictor."""
        # Reseed par round pour reproductibilite.
        if self._search_rng_seed_base is not None:
            logical_round = self._set_logical_round(server_round, config)
            self._search_rng = random.Random(
                self._search_rng_seed_base + int(logical_round) * 1009)
        widths = select_submodels_for_all_workers(
            self.predictor_trainer,
            self.hardware_per_pid,
            num_clients=self.num_clients,
            search_times=self.search_times,
            candidate_widths=self.candidate_widths,
            rng=self._search_rng,
        )
        # Flower n'accepte que des primitives serializables -> floats.
        config["cfl-widths-per-pid"] = [float(w) for w in widths]
        self.last_widths = list(widths)
        # Distribution pour diagnostic
        dist = {w: 0 for w in self.candidate_widths}
        for w in widths:
            closest = min(self.candidate_widths, key=lambda c: abs(c - w))
            dist[closest] = dist.get(closest, 0) + 1
        self.last_width_dist = dist

        msgs = super().configure_train(server_round, arrays, config, grid)
        self._apply_cfl_downlink_accounting(len(msgs))
        return msgs

    def aggregate_train(self, server_round, replies):
        """Zero-pad expansion + moyenne canal-par-canal, puis entrainement
        du predictor avec les (width, acc) du round."""
        replies_list = list(replies)
        valid, _ = self._check_and_log_replies(replies_list, is_train=True)
        contents = [msg.content for msg in valid]
        self._refresh_direct_downlink_bytes(contents)

        # Mesure uplink (bytes reels transmis : varient selon width !)
        raw = sum(self._content_bytes(c) for c in self._uploaded_contents(contents))
        self.last_uplink_bytes = int(raw * self.uplink_size_ratio)
        self.last_technical_uplink_bytes = self.last_uplink_bytes
        self.last_lan_uplink_bytes = 0

        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents or self._last_global_arrays is None:
            return self._last_global_arrays, metrics

        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._last_global_arrays, metrics

        parent_sd = self._last_global_arrays.to_torch_state_dict()

        # pour chaque content : (sub_sd, width, n_k, acc_k, pid, tier)
        rows = []
        for c in accepted:
            ar = next(iter(c.array_records.values()))
            mr = next(iter(c.metric_records.values()))
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            if n_k <= 0:
                continue
            w_k = float(mr.get("cfl_width_ratio", 1.0))
            acc_k = float(mr.get("local_eval_acc", 0.0))
            pid = int(mr.get("partition_id", -1))
            tier = int(mr.get("resource_tier", 1))
            sub_sd = ar.to_torch_state_dict()
            rows.append((sub_sd, w_k, n_k, acc_k, pid, tier))

        if not rows:
            return self._last_global_arrays, metrics

        self.last_sampled_pids = [int(pid) for *_rest, pid, _tier in rows
                                  if int(pid) >= 0]

        # Expansion + moyenne canal-par-canal :
        #   weighted_sum[name] = sum_k (n_k * expanded_sd_k[name] * mask_k[name])
        #   weight_sum[name]   = sum_k (n_k * mask_k[name])
        weighted_sum: Dict[str, torch.Tensor] = {}
        weight_sum: Dict[str, torch.Tensor] = {}

        # Cache des masques par width (pour eviter de recalculer N fois)
        mask_cache: Dict[float, Dict[str, torch.Tensor]] = {}

        for sub_sd, w_k, n_k, _acc, _pid, _tier in rows:
            if w_k not in mask_cache:
                mask_cache[w_k] = active_channel_mask(
                    parent_sd, w_k, self.model_name)
            mask = mask_cache[w_k]

            # Expand le sub_sd a la shape parent (zero-pad).
            expanded = submodel_to_parent_zero_pad(sub_sd, parent_sd)

            # Accumule la somme ponderee et la somme des poids.
            for name, parent_t in parent_sd.items():
                if not parent_t.is_floating_point():
                    continue
                e = expanded[name]
                m = mask[name].to(parent_t.device)
                contrib = e * m * float(n_k)
                w_contrib = m * float(n_k)
                if name in weighted_sum:
                    weighted_sum[name] = weighted_sum[name] + contrib
                    weight_sum[name] = weight_sum[name] + w_contrib
                else:
                    weighted_sum[name] = contrib
                    weight_sum[name] = w_contrib

        # new_w = weighted_sum / weight_sum ; les positions sans aucun client
        # actif conservent la valeur du parent.
        new_sd = {}
        for name, parent_t in parent_sd.items():
            if not parent_t.is_floating_point():
                new_sd[name] = parent_t
                continue
            if name not in weighted_sum:
                new_sd[name] = parent_t
                continue
            w_sum = weight_sum[name]
            safe = w_sum.clone()
            zero_mask = (safe == 0)
            safe[zero_mask] = 1.0
            avg = weighted_sum[name] / safe
            new_sd[name] = torch.where(zero_mask, parent_t, avg)

        # Entraine le predictor sur les samples (width, acc) du round.
        for _sd, w_k, _n_k, acc_k, pid, tier in rows:
            if pid >= 0:
                self.hardware_per_pid[pid] = int(tier)
            self.predictor_trainer.add_sample(w_k, acc_k)
        self.predictor_trainer.train_one_epoch()

        return ArrayRecord(new_sd), metrics
