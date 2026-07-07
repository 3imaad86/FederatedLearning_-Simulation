"""QFedAvgStrategy (Li, Sanjabi, Smith 2019) : fairness par ponderation F_k^q.

Pour chaque client k :
  F_k = loss du modele global sur sa partition (metric `f_k`)
  Delta_k = L * (w_global - w_local_k)
  weighted_delta_k = F_k^q * Delta_k
  h_k = q * F_k^(q-1) * ||Delta_k||^2 + L * F_k^q
Update serveur : w_new = w_global - sum(weighted_delta_k) / sum(h_k)

q grand favorise les clients a haute loss (fairness worst-case) ;
q = 0 donne une moyenne uniforme.
"""

import torch
from flwr.app import ArrayRecord

from fl_common.strategy import FedAvgStrategy, _is_dropped


class QFedAvgStrategy(FedAvgStrategy):
    """q-FedAvg : ponderation par F_k^q (Li-Sanjabi-Smith 2019)."""

    def __init__(self, *args, q: float = 1.0, L: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        if float(q) < 0.0:
            raise ValueError(f"q-FedAvg exige q >= 0 (recu q={q})")
        if float(L) <= 0.0:
            raise ValueError(f"q-FedAvg exige L > 0 (recu L={L})")
        self.q = float(q)
        self.L = float(L)
        # diagnostics exposes par round pour le tail/CSV
        self.last_qfedavg_F_mean = 0.0
        self.last_qfedavg_F_min = 0.0
        self.last_qfedavg_F_max = 0.0
        self.last_qfedavg_step_norm = 0.0

    @staticmethod
    def _safe_pow(base: float, exp: float, eps: float = 1e-10) -> float:
        """F^exp avec clamp pour eviter 0^(neg) = inf et NaN."""
        b = max(float(base), eps)
        return b ** exp

    def aggregate_train(self, server_round, replies):
        """Update q-FedAvg : Delta_k, weighted_delta, h_k, puis sum-aggregation."""
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        self._refresh_direct_downlink_bytes(contents)

        # Uplink bytes (poids complets, FedAvg-equivalent + metric f_k negligeable).
        raw = sum(self._content_bytes(c) for c in self._uploaded_contents(contents))
        self.last_uplink_bytes = int(raw * self.uplink_size_ratio)
        self.last_technical_uplink_bytes = self.last_uplink_bytes
        self.last_lan_uplink_bytes = 0

        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._last_global_arrays, metrics
        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted or self._last_global_arrays is None:
            return self._last_global_arrays, metrics

        global_sd = self._last_global_arrays.to_torch_state_dict()

        # 1) Extrait (w_local_k, F_k, n_k) de chaque client.
        rows = []
        for c in accepted:
            ar = next(iter(c.array_records.values()))
            mr = next(iter(c.metric_records.values()))
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            f_k = float(mr.get("f_k", 0.0))
            if n_k <= 0:
                continue
            local_sd = ar.to_torch_state_dict()
            rows.append((local_sd, f_k, n_k))

        if not rows:
            return self._last_global_arrays, metrics

        # 2) Calcule par client : Delta_k, weighted_delta_k, h_k.
        #    Delta_k = L * (w_global - w_local)
        #    ||Delta_k||^2 = L^2 * sum_param ||w_global - w_local||^2
        L = self.L
        q = self.q

        # Initialise les accumulateurs sum_weighted_delta et sum_h.
        sum_h = 0.0
        sum_weighted_delta = {
            name: torch.zeros_like(t) for name, t in global_sd.items()
            if t.is_floating_point()
        }
        f_values = []  # pour diagnostics

        for local_sd, f_k, _n_k in rows:
            f_values.append(f_k)
            # ||Delta_k||^2 = L^2 * sum (w_g - w_l)^2
            sq_norm = 0.0
            diffs = {}
            for name in sum_weighted_delta:
                if name not in local_sd:
                    continue
                w_g = global_sd[name].to(local_sd[name].device).float()
                w_l = local_sd[name].float()
                diff = w_g - w_l
                diffs[name] = diff
                sq_norm += float((diff * diff).sum().item())
            sq_norm *= (L * L)

            # F_k^q et F_k^(q-1) (avec clamp 0^neg -> eps^neg)
            f_q = self._safe_pow(f_k, q)
            f_qm1 = self._safe_pow(f_k, q - 1.0) if q != 1.0 else 1.0

            # h_k = q * F^(q-1) * ||Delta||^2 + L * F^q
            h_k = q * f_qm1 * sq_norm + L * f_q
            sum_h += h_k

            # weighted_delta_k = F_k^q * Delta_k = F_k^q * L * diff
            scale = f_q * L
            for name, diff in diffs.items():
                sum_weighted_delta[name] = (
                    sum_weighted_delta[name].to(diff.device)
                    + diff * scale
                )

        # 3) Update : w_new = w_global - sum_weighted_delta / sum_h
        if sum_h <= 0.0:
            # Cas degenere (rare, F_k tres petit + q tres grand) : log warning,
            # garde le modele global pour ne pas crash.
            print(f"[q-FedAvg] WARN: sum_h={sum_h} <= 0 au round {server_round}. "
                  f"q={q}, L={L}, F_k_values={f_values}. Modele global conserve.")
            return self._last_global_arrays, metrics

        new_sd = {}
        step_norm_sq = 0.0
        for name, w_g in global_sd.items():
            if not w_g.is_floating_point():
                new_sd[name] = w_g
                continue
            if name not in sum_weighted_delta:
                new_sd[name] = w_g
                continue
            step = sum_weighted_delta[name].to(w_g.device) / sum_h
            new_sd[name] = w_g - step
            step_norm_sq += float((step * step).sum().item())

        # Diagnostics
        if f_values:
            self.last_qfedavg_F_mean = sum(f_values) / len(f_values)
            self.last_qfedavg_F_min = min(f_values)
            self.last_qfedavg_F_max = max(f_values)
        self.last_qfedavg_step_norm = step_norm_sq ** 0.5

        return ArrayRecord(new_sd), metrics
