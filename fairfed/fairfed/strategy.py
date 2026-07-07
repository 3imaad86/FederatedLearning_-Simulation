"""FairFedStrategy (Ezzeldin et al. 2023) : reponderation vers le consensus.

Chaque round :
  Delta_k = |F_k - F_global|, puis
  omega_k = omega_k_precedent - beta * (Delta_k - mean_Delta), normalise.
Les clients proches du consensus pesent plus (l'inverse de q-FedAvg qui
favorise les clients en difficulte).

Adaptation CIFAR-10 : pas d'attribut sensible, donc F_k = train loss du
modele global sur la partition locale. L'EMA (ema_alpha < 1) et le clamp
(max_shift_ratio > 0) sont des extensions optionnelles hors papier.
"""

from flwr.app import ArrayRecord

from fl_common.strategy import FedAvgStrategy, _is_dropped


class FairFedStrategy(FedAvgStrategy):
    """FairFed (Ezzeldin 2023, adapte CIFAR-10) : reponderation vers consensus."""

    def __init__(
        self,
        *args,
        beta: float = 0.1,
        ema_alpha: float = 0.5,
        max_shift_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if float(beta) < 0.0:
            raise ValueError(f"FairFed exige beta >= 0 (recu beta={beta})")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(
                f"FairFed exige 0 < ema_alpha <= 1 (recu {ema_alpha})")
        if float(max_shift_ratio) < 0.0:
            raise ValueError(
                f"FairFed exige max_shift_ratio >= 0 (recu {max_shift_ratio})")

        self.beta = float(beta)
        self.ema_alpha = float(ema_alpha)
        self.max_shift_ratio = float(max_shift_ratio)
        # poids w_k persiste par pid entre rounds (omega^(t-1) du papier)
        self._w_ema = {}
        # diagnostics
        self.last_fairfed_F_mean = 0.0
        self.last_fairfed_F_min = 0.0
        self.last_fairfed_F_max = 0.0
        self.last_fairfed_F_global = 0.0    # consensus
        self.last_fairfed_w_min = 0.0
        self.last_fairfed_w_max = 0.0
        self.last_fairfed_w_var = 0.0       # dispersion finale des poids

    def aggregate_train(self, server_round, replies):
        """Update FairFed : reponderation EMA + moyenne ponderee."""
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        self._refresh_direct_downlink_bytes(contents)

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

        # extrait (w_local_k, F_k, n_k, pid) de chaque client
        rows = []
        for c in accepted:
            ar = next(iter(c.array_records.values()))
            mr = next(iter(c.metric_records.values()))
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            f_k = float(mr.get("f_k", 0.0))
            pid = int(mr.get("partition_id", -1))
            if n_k <= 0:
                continue
            local_sd = ar.to_torch_state_dict()
            rows.append((local_sd, f_k, n_k, pid))

        if not rows:
            return self._last_global_arrays, metrics

        # p_k = n_k / sum(n) (poids FedAvg de base)
        total_n = sum(n for _, _, n, _ in rows)
        if total_n <= 0:
            return self._last_global_arrays, metrics
        p_list = [n / total_n for _, _, n, _ in rows]
        f_list = [f for _, f, _, _ in rows]
        pid_list = [pid for _, _, _, pid in rows]

        # F_global = sum_k p_k * F_k (consensus de loss)
        f_global = sum(p * f for p, f in zip(p_list, f_list))

        # Delta_k = |F_k - F_global|, mean_Delta = moyenne des Delta_k
        delta_list = [abs(f - f_global) for f in f_list]
        mean_delta = sum(delta_list) / len(delta_list)

        # omega_bar_k = omega_precedent_k - beta * (Delta_k - mean_Delta)
        # (au round 1, omega_precedent = p_k).
        prev_w_list = [
            float(self._w_ema.get(pid, p))
            for pid, p in zip(pid_list, p_list)
        ]
        raw_w_list = [
            prev - self.beta * (d - mean_delta)
            for prev, d in zip(prev_w_list, delta_list)
        ]
        # Un Delta tres grand peut donner un poids negatif -> clamp >= 0.
        raw_w_list = [max(0.0, w) for w in raw_w_list]

        # Extension optionnelle : clamp autour de p_k pour limiter les
        # oscillations (desactive par defaut).
        if self.max_shift_ratio > 0.0:
            clamped = []
            for raw_w, p in zip(raw_w_list, p_list):
                lo = max(0.0, p * (1.0 - self.max_shift_ratio))
                hi = p * (1.0 + self.max_shift_ratio)
                clamped.append(min(max(raw_w, lo), hi))
            raw_w_list = clamped

        # Extension optionnelle : EMA avec le round precedent
        # (ema_alpha = 1.0 par defaut = pas de lissage).
        if self.ema_alpha < 1.0:
            final_w_list = []
            for raw_w, prev in zip(raw_w_list, prev_w_list):
                final = self.ema_alpha * raw_w + (1.0 - self.ema_alpha) * prev
                final_w_list.append(final)
        else:
            final_w_list = list(raw_w_list)

        # Normalise (somme = 1).
        total_w = sum(final_w_list)
        if total_w <= 0.0:
            # Tous les poids sont 0 -> fallback sur p_k.
            print(f"[FairFed] WARN: sum(w)=0 au round {server_round}. "
                  "Fallback sur p_k = n_k/sum(n).")
            final_w_list = list(p_list)
            total_w = sum(final_w_list)
        norm_w_list = [w / total_w for w in final_w_list]

        # persiste le poids normalise pour le round suivant (= omega^(t-1))
        for pid, w in zip(pid_list, norm_w_list):
            if pid >= 0:
                self._w_ema[pid] = float(w)

        # aggregation ponderee : w_new = sum_k (norm_w_k * w_local_k)
        global_sd = self._last_global_arrays.to_torch_state_dict()
        new_sd = {}
        for name, ref_t in global_sd.items():
            if not ref_t.is_floating_point():
                new_sd[name] = ref_t
                continue
            acc = None
            for (local_sd, _f, _n, _pid), w in zip(rows, norm_w_list):
                if name not in local_sd:
                    continue
                term = local_sd[name].to(ref_t.device).float() * float(w)
                acc = term if acc is None else acc + term
            new_sd[name] = acc if acc is not None else ref_t

        # Diagnostics
        self.last_fairfed_F_mean = sum(f_list) / len(f_list)
        self.last_fairfed_F_min = min(f_list)
        self.last_fairfed_F_max = max(f_list)
        self.last_fairfed_F_global = float(f_global)
        self.last_fairfed_w_min = float(min(norm_w_list))
        self.last_fairfed_w_max = float(max(norm_w_list))
        w_mean = sum(norm_w_list) / len(norm_w_list)
        self.last_fairfed_w_var = float(
            sum((w - w_mean) ** 2 for w in norm_w_list) / len(norm_w_list))

        return ArrayRecord(new_sd), metrics
