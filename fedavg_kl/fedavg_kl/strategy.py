"""FedAvg+KL : agregation FedAvg des poids + distillation KL via proxy.

Variante hybride ou la distillation regularise FedAvg (inspiree de la
partie proxy/KMeans-DRE d'EdgeFD, Liu et al. 2025).

Protocole par round :
  1. Le serveur envoie les poids globaux + les logits agreges du round
     precedent + le masque de validite + les indices proxy du round.
  2. Les clients entrainent (CE + KL sur le proxy), calculent leurs logits
     sur le proxy, les filtrent via KMeans-DRE, et renvoient
     (poids, logits filtres, masque).
  3. Le serveur agrege les poids (FedAvg) et les logits position par
     position selon les masques.
"""

import torch
from flwr.app import ArrayRecord

from fl_common.strategy import (
    FedAvgStrategy,
    _array_record_bytes,
    _is_dropped,
)


# Prefixes utilises pour packer dans un seul ArrayRecord :
#   __agg_logits__     : logits agreges r-1 (server -> client)
#   __agg_valid_mask__ : positions du proxy avec un signal teacher
#   __logits__         : logits client (client -> server)
#   __id_mask__        : masque ID/OOD client (client -> server)
#   __proxy_idx__      : indices proxy des logits client compacts
LOGITS_PREFIX_AGG = "__agg_logits__"
LOGITS_VALID_PREFIX_AGG = "__agg_valid_mask__"
LOGITS_PREFIX_CLI = "__logits__"
MASK_PREFIX_CLI = "__id_mask__"
INDEX_PREFIX_CLI = "__proxy_idx__"


def _split_client_reply(content):
    """Decode (model_sd, logits, mask, indices) depuis le content du client."""
    ar = next(iter(content.array_records.values()))
    full = ar.to_torch_state_dict()
    model_sd, logits, mask, indices = {}, None, None, None
    for k, v in full.items():
        if k.startswith(LOGITS_PREFIX_CLI):
            logits = v
        elif k.startswith(MASK_PREFIX_CLI):
            mask = v
        elif k.startswith(INDEX_PREFIX_CLI):
            indices = v
        else:
            model_sd[k] = v
    return model_sd, logits, mask, indices


class FedAvgKLStrategy(FedAvgStrategy):
    """FedAvg+KL : agregation FedAvg des poids + agregation des logits proxy.

    Les logits agreges sont rebroadcastes au round suivant comme teacher de
    distillation. Le masque de validite evite de distiller vers une
    distribution uniforme aux positions du proxy sans contributeur.
    """

    def __init__(
        self,
        *args,
        proxy_size: int = 2000,
        num_classes: int = 10,
        distill_lambda: float = 0.5,
        distill_temperature: float = 4.0,
        proxy_per_round: int = 500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.proxy_size = max(1, int(proxy_size))
        self.num_classes = max(2, int(num_classes))
        self.distill_lambda = float(distill_lambda)
        self.distill_T = float(distill_temperature)
        self.proxy_per_round = max(0, int(proxy_per_round))
        # Logits agreges du round precedent (zeros = distillation neutre au round 1).
        self._agg_logits = torch.zeros(
            (self.proxy_size, self.num_classes), dtype=torch.float32)
        # Masque sticky des positions avec un signal teacher reel.
        self._agg_valid_mask = torch.zeros(
            (self.proxy_size,), dtype=torch.float32)
        # Diagnostics
        self.last_id_ratio = 0.0       # fraction moyenne de samples ID
        self.last_logits_active = 0    # nb de positions proxy avec signal
        self.last_logits_bytes = 0     # surcout comm de la distillation
        self.last_proxy_indices = []

    @staticmethod
    def _is_logits_key(name):
        """True si la cle correspond a un payload logits/mask/index."""
        return (
            name == LOGITS_PREFIX_AGG
            or name == LOGITS_VALID_PREFIX_AGG
            or name.startswith(LOGITS_PREFIX_CLI)
            or name.startswith(MASK_PREFIX_CLI)
            or name.startswith(INDEX_PREFIX_CLI)
        )

    def _record_bytes(self, arrays):
        """Bytes d'un ArrayRecord : le modele est sim-scale par sim_model_mb,
        mais les logits/mask/index gardent leur taille intrinseque."""
        sd = arrays.to_torch_state_dict()
        model_bytes = 0
        logits_bytes = 0
        for name, t in sd.items():
            n = t.numel() * t.element_size()
            if self._is_logits_key(name):
                logits_bytes += n
            else:
                model_bytes += n
        return int(model_bytes * self._sim_scale()) + logits_bytes

    def _content_bytes(self, content):
        """Bytes d'un content client : meme logique que _record_bytes."""
        total = 0
        for ar in content.array_records.values():
            total += self._record_bytes(ar)
        return total

    def _sample_proxy_indices(self, server_round, config):
        """Echantillonne `proxy_per_round` indices du proxy (seed deterministe
        pour que tous les clients voient les memes indices au meme round)."""
        if self.proxy_per_round <= 0 or self.proxy_per_round >= self.proxy_size:
            return list(range(self.proxy_size))
        seed_base = int(config.get("seed", 42)) if config else 42
        if seed_base < 0:
            seed_base = 42
        logical_round = self._round(server_round)
        gen = torch.Generator()
        gen.manual_seed(seed_base * 1009 + int(logical_round) * 31337)
        idx = torch.randperm(self.proxy_size, generator=gen).tolist()
        return idx[: self.proxy_per_round]

    def configure_train(self, server_round, arrays, config, grid):
        """Pack w_global + agg_logits + valid_mask dans un seul ArrayRecord."""
        self._set_logical_round(server_round, config)
        proxy_indices = self._sample_proxy_indices(server_round, config)
        self.last_proxy_indices = proxy_indices
        # ConfigRecord ne prend que des primitives -> liste d'int.
        config["fedavgkl-proxy-indices"] = [int(i) for i in proxy_indices]
        config["fedavgkl-proxy-size"] = int(self.proxy_size)
        config["fedavgkl-distill-lambda"] = float(self.distill_lambda)
        config["fedavgkl-distill-temperature"] = float(self.distill_T)

        w_sd = arrays.to_torch_state_dict()
        combined = dict(w_sd)
        combined[LOGITS_PREFIX_AGG] = self._agg_logits.detach().clone()
        combined[LOGITS_VALID_PREFIX_AGG] = self._agg_valid_mask.detach().clone()
        msgs = super().configure_train(
            server_round, ArrayRecord(combined), config, grid)
        # _last_global_arrays doit pointer vers les poids purs (sans logits)
        # pour le fallback en cas de drop total.
        self._last_global_arrays = arrays
        return msgs

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        self._refresh_direct_downlink_bytes(contents)

        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._last_global_arrays, metrics

        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._last_global_arrays, metrics

        # 1) Decode chaque reply (model_sd, logits, mask) + n_k.
        model_replies = []        # liste (sd, n_k) pour FedAvg
        logits_rows = []          # liste (logits, mask, indices, n_k)
        for c in accepted:
            model_sd, logits, mask, indices = _split_client_reply(c)
            mr = next(iter(c.metric_records.values()))
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            if n_k <= 0:
                continue
            if model_sd:
                model_replies.append((model_sd, n_k))
            if logits is not None and mask is not None:
                logits_rows.append((logits, mask, indices, n_k))

        # 2) Agrege les poids FedAvg-style (moyenne ponderee par n_k).
        if not model_replies:
            new_arrays = self._last_global_arrays
        else:
            total_n = sum(n for _, n in model_replies)
            ref_sd = model_replies[0][0]
            new_sd = {}
            for k, ref_t in ref_sd.items():
                if not ref_t.is_floating_point():
                    new_sd[k] = ref_t
                    continue
                acc = None
                for sd, n in model_replies:
                    if k not in sd:
                        continue
                    term = sd[k].to(ref_t.device) * (float(n) / total_n)
                    acc = term if acc is None else acc + term
                new_sd[k] = acc if acc is not None else ref_t
            new_arrays = ArrayRecord(new_sd)

        # 3) Mesure uplink : modele complet + logits + masque.
        raw_total = sum(
            self._content_bytes(c) for c in self._uploaded_contents(contents))
        self.last_uplink_bytes = int(raw_total * self.uplink_size_ratio)
        self.last_technical_uplink_bytes = self.last_uplink_bytes
        self.last_lan_uplink_bytes = 0
        # Diagnostic : taille des logits + masque seuls.
        logits_bytes = 0
        for logits, mask, indices, _ in logits_rows:
            logits_bytes += int(logits.numel() * logits.element_size())
            logits_bytes += int(mask.numel() * mask.element_size())
            if indices is not None:
                logits_bytes += int(indices.numel() * indices.element_size())
        self.last_logits_bytes = logits_bytes

        # 4) Agrege les logits position par position selon le masque :
        #    agg_logits[i, c] = sum_k n_k * mask_k[i] * logits_k[i, c]
        #                       / sum_k n_k * mask_k[i]
        #    Les positions sans contributeur gardent les anciens logits.
        if logits_rows:
            K = int(logits_rows[0][0].size(1))
            weighted_sum = torch.zeros((self.proxy_size, K), dtype=torch.float32)
            weight_sum = torch.zeros((self.proxy_size,), dtype=torch.float32)
            for logits, mask, indices, n_k in logits_rows:
                logits = logits.float()
                mask = mask.float().view(-1)
                if indices is None:
                    # Compat ancienne reply : logits deja a la taille du proxy.
                    idx = torch.arange(int(logits.size(0)), dtype=torch.long)
                else:
                    idx = indices.long().view(-1)
                if int(logits.size(0)) != int(idx.numel()):
                    continue
                idx = idx.clamp(0, self.proxy_size - 1)
                m = mask.view(-1, 1)
                weighted_sum[idx] += logits * m * float(n_k)
                weight_sum[idx] += mask * float(n_k)
            zero_mask = (weight_sum <= 0.0)
            safe = weight_sum.clone()
            safe[zero_mask] = 1.0
            avg = weighted_sum / safe.unsqueeze(1)
            old = self._agg_logits.float()
            self._agg_logits = torch.where(
                zero_mask.unsqueeze(1), old, avg).detach()
            # Une position validee le reste (sticky).
            new_valid = (weight_sum > 0.0).float()
            self._agg_valid_mask = torch.maximum(
                self._agg_valid_mask, new_valid).detach()
            # Diagnostics
            n_active = int((weight_sum > 0).sum().item())
            self.last_logits_active = n_active
            total_id = float(
                sum(float(m.sum().item()) for _, m, _, _ in logits_rows))
            total_proxy = float(
                sum(float(m.numel()) for _, m, _, _ in logits_rows))
            self.last_id_ratio = total_id / total_proxy if total_proxy > 0 else 0.0
        else:
            self.last_logits_active = 0
            self.last_id_ratio = 0.0

        return new_arrays, metrics
