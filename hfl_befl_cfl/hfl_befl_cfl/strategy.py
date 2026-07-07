"""CFLHFLBeflStrategy : RLHierarchicalStrategy (HFL-BEFL) + CFL.

configure_train decide 3 listes par client : width CFL (predictor), tau
BEFL (Lyapunov) et edge_id (RL). aggregate_train agrege les edges en CFL
(width expansion canal-par-canal), le cloud en FedAvg, puis met a jour
BEFL, le predictor CFL et la policy RL.
"""

import random
from typing import Dict, List

import torch

from ._local_imports import ensure_workspace_packages

ensure_workspace_packages()

from cfl.predictor import AccuracyPredictor, AccuracyPredictorTrainer
from cfl.search import select_submodels_for_all_workers
from cfl.submodel import (
    DEFAULT_CANDIDATE_WIDTHS,
    active_channel_mask,
    submodel_to_parent_zero_pad,
)
from fl_common.strategy import _is_dropped, _no_model_transfer

from hfl_befl.server_app import RLHierarchicalStrategy


class CFLHFLBeflStrategy(RLHierarchicalStrategy):
    """RLHierarchicalStrategy + width selection (CFL) + CFL edge aggregation."""

    def __init__(
        self,
        *args,
        cfl_search_times: int = 10,
        cfl_predictor_lr: float = 0.01,
        cfl_predictor_hidden: int = 32,
        cfl_candidate_widths: List[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # etat CFL
        self.cfl_search_times = int(cfl_search_times)
        self.cfl_candidate_widths = list(
            cfl_candidate_widths or DEFAULT_CANDIDATE_WIDTHS)
        self.predictor = AccuracyPredictor(hidden_dim=int(cfl_predictor_hidden))
        self.predictor_trainer = AccuracyPredictorTrainer(
            self.predictor, lr=float(cfl_predictor_lr))
        # tier hardware par pid, alimente au fil des replies
        self.cfl_hardware_per_pid: Dict[int, int] = {}
        # widths attribues ce round (pid -> width)
        self.cfl_widths_per_pid: Dict[int, float] = {}
        # RNG dedie a la recherche CFL, reseed a chaque round
        self._cfl_rng_seed_base = (
            self.seed * 8147 + 23 if self.seed >= 0 else None)
        self._cfl_rng = random.Random(self._cfl_rng_seed_base)
        # diagnostics
        self.last_widths = []
        self.last_width_dist = {}

        # FedStrag + CFL est mal defini (une update stale peut avoir un
        # width d'un ancien round) : on refuse la configuration.
        if self.fedstrag_enabled:
            raise ValueError(
                "HFL-BEFL-CFL ne supporte pas fedstrag-enabled=1: les updates "
                "stale peuvent avoir un width different du round courant, "
                "ce qui rend l'agregation CFL au niveau edge ambigue. "
                "Configure fedstrag-enabled=0 pour cette combinaison."
            )

    def _mean_width_sq(self, contents=None) -> float:
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
            widths = [float(w) for w in (self.last_widths or [])]
        if not widths:
            return 1.0
        return sum(w * w for w in widths) / len(widths)

    def _apply_cfl_lan_downlink_accounting(self, contents):
        """Ajuste le downlink LAN logique a la taille des submodels CFL
        (le payload technique Flower reste le full model packe)."""
        download_count = self._download_count(contents)
        model_bytes = self._base_model_bytes(self._last_global_arrays)
        self.last_lan_downlink_bytes = int(
            model_bytes * download_count * self._mean_width_sq(contents)
            * self.comm_size_ratio
        )

    def _apply_tier_uplink_accounting(self, contents):
        """Recalcule l'uplink LAN avec le ratio compression propre a chaque tier."""
        total = 0.0
        for content in contents or []:
            if _is_dropped(content):
                continue
            try:
                mr = next(iter(content.metric_records.values()))
            except (AttributeError, ValueError, StopIteration):
                continue
            ratio = float(mr.get("uplink_size_ratio", self.uplink_size_ratio))
            total += float(self._content_bytes(content)) * ratio
        self.last_lan_uplink_bytes = int(total)
        self.last_technical_uplink_bytes = self.last_lan_uplink_bytes

    def configure_train(self, server_round, arrays, config, grid):
        """Selection widths CFL, puis le parent injecte tau BEFL + edge RL."""
        # Reseed le RNG CFL a chaque round pour rester reproductible.
        if self._cfl_rng_seed_base is not None:
            logical_round = self._set_logical_round(server_round, config)
            self._cfl_rng = random.Random(
                self._cfl_rng_seed_base + int(logical_round) * 1009)
        # CFL : selection des widths via predictor + latency budget
        widths = select_submodels_for_all_workers(
            self.predictor_trainer,
            self.cfl_hardware_per_pid,
            num_clients=self.num_clients,
            search_times=self.cfl_search_times,
            candidate_widths=self.cfl_candidate_widths,
            rng=self._cfl_rng,
        )
        config["cfl-widths-per-pid"] = [float(w) for w in widths]
        self.last_widths = list(widths)
        self.cfl_widths_per_pid = {
            pid: float(w) for pid, w in enumerate(widths)
        }
        # distribution diagnostique
        dist = {w: 0 for w in self.cfl_candidate_widths}
        for w in widths:
            closest = min(self.cfl_candidate_widths, key=lambda c: abs(c - w))
            dist[closest] = dist.get(closest, 0) + 1
        self.last_width_dist = dist

        return super().configure_train(server_round, arrays, config, grid)

    def _hfl_client_groups(self, logical_round, contents, global_sd):
        """Attache (pid, width) en plus de (sd, n) a chaque row d'edge.

        Le parent ne lit que les 2 premiers elements du tuple, donc on
        reste compatible.
        """
        groups = {}
        for content in contents:
            if _is_dropped(content):
                continue
            ar = next(iter(content.array_records.values()))
            mr = next(iter(content.metric_records.values()))
            n_i = float(mr.get(self.weighted_by_key, 0.0))
            if n_i <= 0:
                continue
            edge_id = int(mr.get("hfl_edge_id", 0)) % self.num_edges
            pid = int(mr.get("partition_id", -1))
            width = float(mr.get("cfl_width_ratio", 1.0))
            sub_sd = ar.to_torch_state_dict()
            groups.setdefault(edge_id, []).append((sub_sd, n_i, pid, width))
        return groups

    def _aggregate_edge_clients(self, edge_id, edge_ref_sd, rows):
        """Agregation CFL au niveau edge : zero-pad expansion + moyenne
        canal-par-canal ponderee par les masques actifs.

        rows : liste de 4-tuples (sd, n_k, pid, width).
        """
        if not rows:
            return edge_ref_sd

        weighted_sum: Dict[str, torch.Tensor] = {}
        weight_sum: Dict[str, torch.Tensor] = {}
        mask_cache: Dict[float, Dict[str, torch.Tensor]] = {}

        for item in rows:
            sub_sd, n_k, _pid, w_k = item[0], item[1], item[2], item[3]
            if n_k <= 0:
                continue
            if w_k not in mask_cache:
                mask_cache[w_k] = active_channel_mask(
                    edge_ref_sd, w_k, self.model_name)
            mask = mask_cache[w_k]
            expanded = submodel_to_parent_zero_pad(sub_sd, edge_ref_sd)

            for name, parent_t in edge_ref_sd.items():
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

        new_sd = {}
        for name, ref_t in edge_ref_sd.items():
            if not ref_t.is_floating_point():
                new_sd[name] = ref_t
                continue
            if name not in weighted_sum:
                new_sd[name] = ref_t
                continue
            w_sum = weight_sum[name]
            safe = w_sum.clone()
            zero_mask = (safe == 0)
            safe[zero_mask] = 1.0
            avg = weighted_sum[name] / safe
            # Positions sans aucun client actif -> garde edge_ref (= parent).
            new_sd[name] = torch.where(zero_mask, ref_t, avg)

        return new_sd

    def aggregate_train(self, server_round, replies):
        """Pipeline HFL-BEFL + entrainement du predictor CFL."""
        replies_list = list(replies)

        # Collecte les samples (w_k, acc_k) pour le predictor et met a jour
        # hardware_per_pid pour le round suivant.
        for msg in replies_list:
            try:
                content = msg.content
                mr = next(iter(content.metric_records.values()))
            except (ValueError, StopIteration, AttributeError):
                continue
            if float(mr.get("dropped", 0.0)) >= 0.5:
                continue
            pid = int(mr.get("partition_id", -1))
            tier = int(mr.get("resource_tier", 1))
            w_k = float(mr.get("cfl_width_ratio", 1.0))
            acc_k = float(mr.get("local_eval_acc", 0.0))
            if pid >= 0:
                self.cfl_hardware_per_pid[pid] = int(tier)
            self.predictor_trainer.add_sample(w_k, acc_k)
        self.predictor_trainer.train_one_epoch()

        # Delegue le reste a HFL-BEFL ; le parent appelle notre
        # _aggregate_edge_clients (CFL) au niveau edge.
        result = super().aggregate_train(server_round, replies_list)

        contents = []
        for msg in replies_list:
            try:
                if getattr(msg, "has_content", lambda: True)() and msg.content is not None:
                    contents.append(msg.content)
            except Exception:
                continue
        self._apply_cfl_lan_downlink_accounting(contents)
        self._apply_tier_uplink_accounting(contents)
        return result
