"""Strategies partagees par les algos FL du repo.

  * FedAvgStrategy       : FedAvg + mesure des bytes transmis chaque round.
  * FedProxStrategy      : FedAvg + capture du prox_term moyen (diagnostic).
  * FedNovaStrategy      : FedNova (rescaling tau_eff cote serveur).
  * ScaffoldStrategy     : SCAFFOLD avec control variates.
  * FedEveStrategy       : fusion momentum serveur / updates clients (Kalman).
  * HierarchicalStrategy : HFL device -> edge -> cloud.
"""

import random

import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from .data import model_size_bytes
from .energy import compute_edge_cloud_energy_j
from .straggler import mbps_transfer_time_s


def _is_dropped(content):
    """True si le client a renvoye un reply 'drop' (panne/deadline)."""
    mr = next(iter(content.metric_records.values()))
    return float(mr.get("dropped", 0.0)) >= 0.5


def _no_model_transfer(content):
    """True si le client n'a recu/envoye aucun modele sur ce round."""
    mr = next(iter(content.metric_records.values()))
    return float(mr.get("no_model_transfer", 0.0)) >= 0.5


def _array_record_bytes(arrays):
    """Taille (octets) d'un ArrayRecord."""
    sd = arrays.to_torch_state_dict()
    return sum(t.numel() * t.element_size() for t in sd.values())


def _content_bytes(content):
    """Taille totale des ArrayRecords dans un content."""
    return sum(_array_record_bytes(ar) for ar in content.array_records.values())


class FedAvgStrategy(FedAvg):
    """FedAvg + mesure des bytes reels transmis chaque round (downlink + uplink).

    Base de tous les algos du repo : comptabilise les communications
    (compatible comm_size_ratio et sim_model_mb), ignore les replies
    marquees dropped=1, et separe les bytes WAN (cloud) des bytes LAN (edge).
    """

    def __init__(self, *args, comm_size_ratio=1.0, uplink_size_ratio=None,
                 sim_model_mb=0.0, model_name="net", **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_arrays = None
        self._logical_server_round = 0
        # La compression quant/sparse est cote client : elle ne reduit que
        # l'uplink, d'ou un ratio uplink separe.
        self.comm_size_ratio = float(comm_size_ratio)
        self.uplink_size_ratio = (
            self.comm_size_ratio if uplink_size_ratio is None
            else float(uplink_size_ratio)
        )
        # Taille modele simulee en MB (0 = vraie taille).
        self.sim_model_mb = float(sim_model_mb)
        self.model_name = str(model_name)
        # Bytes WAN du dernier round (pour HFL : seulement edges <-> cloud).
        self.last_downlink_bytes = 0
        self.last_uplink_bytes = 0
        # Bytes LAN (edge-local). Toujours 0 sauf pour HFL.
        self.last_lan_downlink_bytes = 0
        self.last_lan_uplink_bytes = 0
        # Payload Flower reellement transporte par la simulation.
        self.last_technical_downlink_bytes = 0
        self.last_technical_uplink_bytes = 0
        self._last_train_msg_count = 0
        self._last_downlink_record_bytes = 0

    def _sim_scale(self):
        """Facteur qui transforme la vraie taille du modele en taille simulee."""
        if self.sim_model_mb <= 0:
            return 1.0
        return (self.sim_model_mb * 1024.0 * 1024.0) / max(
            1, model_size_bytes(self.model_name))

    def _record_bytes(self, arrays):
        """Taille effective d'un ArrayRecord, en respectant sim-model-mb."""
        return int(_array_record_bytes(arrays) * self._sim_scale())

    def _content_bytes(self, content):
        """Taille effective d'un content Flower, en respectant sim-model-mb."""
        return int(_content_bytes(content) * self._sim_scale())

    def _base_model_bytes(self, arrays=None):
        """Taille effective d'un seul modele non compresse."""
        if self.sim_model_mb > 0:
            return int(self.sim_model_mb * 1024.0 * 1024.0)
        if arrays is not None:
            return _array_record_bytes(arrays)
        return model_size_bytes(self.model_name)

    def _uploaded_contents(self, contents):
        """Contents qui ont reellement envoye un modele sur le lien montant."""
        return [c for c in contents if not _is_dropped(c)]

    def _download_count(self, contents):
        """Nombre de clients ayant reellement recu le modele complet."""
        skipped = sum(1 for c in contents if _no_model_transfer(c))
        return max(0, int(self._last_train_msg_count) - skipped)

    def _refresh_direct_downlink_bytes(self, contents):
        """Ajuste le downlink direct apres observation des drops precoces."""
        self.last_downlink_bytes = int(
            self._last_downlink_record_bytes
            * self._download_count(contents)
            * self.comm_size_ratio
        )
        self.last_technical_downlink_bytes = self.last_downlink_bytes

    def _set_logical_round(self, server_round, config=None):
        """Memorise le round applicatif envoye dans train_config.

        Le runner appelle Flower round par round (num_rounds=1), donc le
        server_round interne de Flower revient a 1 a chaque appel. Le round
        logique du run est transporte dans config["round"].
        """
        logical_round = server_round
        if config is not None:
            try:
                logical_round = int(config.get("round", server_round))
            except (TypeError, ValueError):
                logical_round = server_round
        self._logical_server_round = int(logical_round)
        return self._logical_server_round

    def _round(self, server_round):
        """Retourne le round logique courant, fallback sur Flower server_round."""
        return int(self._logical_server_round or server_round)

    def configure_train(self, server_round, arrays, config, grid):
        self._set_logical_round(server_round, config)
        # Garde une ref aux poids globaux (fallback si aucun reply valide)
        self._last_global_arrays = arrays
        msgs = super().configure_train(server_round, arrays, config, grid)
        # Downlink = cloud -> chaque client, avec comm_size_ratio applique ici.
        raw = self._record_bytes(arrays) * len(msgs)
        self._last_train_msg_count = len(msgs)
        self._last_downlink_record_bytes = self._record_bytes(arrays)
        self.last_downlink_bytes = int(raw * self.comm_size_ratio)
        self.last_technical_downlink_bytes = self.last_downlink_bytes
        self.last_technical_uplink_bytes = 0
        self.last_lan_downlink_bytes = 0
        return msgs

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        self._refresh_direct_downlink_bytes(contents)
        # Uplink : les drops n'uploadent pas de modele.
        raw = sum(self._content_bytes(c) for c in self._uploaded_contents(contents))
        self.last_uplink_bytes = int(raw * self.uplink_size_ratio)
        self.last_technical_uplink_bytes = self.last_uplink_bytes
        self.last_lan_uplink_bytes = 0

        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        if not contents:
            return self._last_global_arrays, metrics

        accepted = [c for c in contents if not _is_dropped(c)]
        arrays = (aggregate_arrayrecords(accepted, self.weighted_by_key)
                  if accepted else self._last_global_arrays)
        return arrays, metrics


class FedProxStrategy(FedAvgStrategy):
    """FedProx serveur = FedAvgStrategy + capture du prox_term moyen.

    L'agregation reste FedAvg (le terme proximal est cote client). On expose
    juste `last_prox_term_mean` pour verifier que mu est bien calibre.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prox_term_mean = 0.0

    def aggregate_train(self, server_round, replies):
        # On materialise la liste pour pouvoir la parcourir 2x.
        replies = list(replies)
        total_n = 0.0
        weighted_sum = 0.0
        for msg in replies:
            try:
                mr = next(iter(msg.content.metric_records.values()))
            except (AttributeError, ValueError, StopIteration):
                continue
            if float(mr.get("dropped", 0.0)) >= 0.5:
                continue
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            pt = float(mr.get("prox_term", 0.0))
            if n_k > 0:
                weighted_sum += n_k * pt
                total_n += n_k
        self.last_prox_term_mean = (
            weighted_sum / total_n if total_n > 0 else 0.0
        )
        return super().aggregate_train(server_round, replies)


def _extract_delta_tau_n(content, global_sd, weighted_by_key="num-examples"):
    """Pour un client : retourne (delta_dict, tau_i, n_i) ou None si invalide."""
    ar = next(iter(content.array_records.values()))
    mr = next(iter(content.metric_records.values()))
    tau_i = float(mr.get("tau_i", 0.0))
    n_i = float(mr.get(weighted_by_key, 0.0))
    if tau_i <= 0 or n_i <= 0:
        return None
    local_sd = ar.to_torch_state_dict()
    delta = {
        k: v - global_sd[k].to(v.device)
        for k, v in local_sd.items() if v.is_floating_point()
    }
    return delta, tau_i, n_i


def _apply_fednova_update(global_sd, deltas, taus, ns):
    """w_new = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i)."""
    total_n = sum(ns)
    tau_eff = sum((n / total_n) * t for n, t in zip(ns, taus))

    new_sd = {}
    for k, w_global in global_sd.items():
        if not w_global.is_floating_point():
            new_sd[k] = w_global
            continue
        acc = None
        for delta, tau_i, n_i in zip(deltas, taus, ns):
            if k not in delta:
                continue
            term = delta[k].to(w_global.device) * ((n_i / total_n) / tau_i)
            acc = term if acc is None else acc + term
        if acc is None:
            new_sd[k] = w_global
        else:
            new_sd[k] = w_global + tau_eff * acc
    return new_sd, tau_eff


class FedNovaStrategy(FedAvgStrategy):
    """FedNova : tau_eff calcule + applique cote serveur."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tau_eff = 0.0
        # min/max/ratio des tau pour rendre l'heterogeneite visible dans les logs.
        self.last_tau_min = 0.0
        self.last_tau_max = 0.0
        self.last_tau_ratio = 1.0

    def aggregate_train(self, server_round, replies):
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
        if self._last_global_arrays is None:
            return None, metrics

        global_sd = self._last_global_arrays.to_torch_state_dict()

        # Si tous les clients ont droppe, on garde le modele global inchange.
        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._last_global_arrays, metrics
        triples = [
            _extract_delta_tau_n(c, global_sd, self.weighted_by_key)
            for c in accepted
        ]
        triples = [t for t in triples if t is not None]

        if not triples:
            print(f"[fednova] WARN: aucun client utilisable au round "
                  f"{server_round} ({len(accepted)} acceptes, tous avec "
                  f"tau_i<=0 ou n_i<=0). Modele global conserve.")
            return self._last_global_arrays, metrics

        deltas, taus, ns = zip(*triples)
        new_sd, tau_eff = _apply_fednova_update(global_sd, deltas, taus, ns)
        if tau_eff <= 0.0:
            print(f"[fednova] WARN: tau_eff={tau_eff} <= 0 au round "
                  f"{server_round} avec {len(triples)} clients utiles. "
                  "Modele global conserve.")
            return self._last_global_arrays, metrics
        self.last_tau_eff = tau_eff
        tau_min = float(min(taus))
        tau_max = float(max(taus))
        self.last_tau_min = tau_min
        self.last_tau_max = tau_max
        self.last_tau_ratio = (tau_max / tau_min) if tau_min > 0 else 1.0
        return ArrayRecord(new_sd), metrics


# Prefixes pour packer plusieurs arrays dans un seul ArrayRecord :
#   __cg__ = control global (server -> client)
#   __dc__ = delta control  (client -> server)
CG_PREFIX = "__cg__"
DC_PREFIX = "__dc__"


class ScaffoldStrategy(FedAvgStrategy):
    """SCAFFOLD : FedAvg + control variates (Karimireddy et al. 2020).

    Le serveur maintient w_global et c_global ; chaque client garde son
    c_local. w_global et c_global sont packes dans un seul ArrayRecord
    (prefixes __cg__ / __dc__). Update serveur :
      w_new = w + eta_g * mean(delta_y),  c_new = c + (|S|/N) * mean(delta_c)
    """

    def __init__(self, *args, num_clients_total=None, server_lr=1.0,
                 param_names=None, weighted_aggregation=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._c_global_sd = None     # init au 1er configure_train
        self._w_global_arrays = None  # version SANS c_global (pour fallback)
        # Diagnostics par round : norme de c_global et du delta_c agrege.
        self.last_c_global_norm = 0.0
        self.last_delta_c_norm = 0.0
        # Total clients dans la federation (pour le facteur |S|/N).
        self.num_clients_total = num_clients_total
        # LR serveur (eta_g dans le papier).
        self.server_lr = float(server_lr)
        # Option : moyenne ponderee par n_k au lieu d'uniforme.
        self.weighted_aggregation = bool(int(weighted_aggregation))
        # Whitelist optionnelle des noms de parametres, pour que c_global
        # garde exactement les memes cles que les params entraines.
        self.param_names = set(param_names) if param_names else None

    def _init_c_global(self, w_state_dict):
        """Initialise c_global a zeros (meme shape que les params)."""
        if self.param_names is not None:
            self._c_global_sd = {
                name: torch.zeros_like(t)
                for name, t in w_state_dict.items()
                if name in self.param_names
            }
        else:
            self._c_global_sd = {
                name: torch.zeros_like(t)
                for name, t in w_state_dict.items()
                if t.is_floating_point()
            }

    def configure_train(self, server_round, arrays, config, grid):
        """Pack w_global et c_global dans un seul ArrayRecord."""
        w_sd = arrays.to_torch_state_dict()
        if self._c_global_sd is None:
            self._init_c_global(w_sd)
        self._w_global_arrays = arrays

        combined = dict(w_sd)
        for name, t in self._c_global_sd.items():
            combined[f"{CG_PREFIX}{name}"] = t
        return super().configure_train(server_round, ArrayRecord(combined), config, grid)

    @staticmethod
    def _split_y_and_dc(content):
        """Decode un content packe en (y_sd, dc_sd)."""
        ar = next(iter(content.array_records.values()))
        full_sd = ar.to_torch_state_dict()
        y_sd, dc_sd = {}, {}
        for k, v in full_sd.items():
            if k.startswith(DC_PREFIX):
                dc_sd[k[len(DC_PREFIX):]] = v
            else:
                y_sd[k] = v
        return y_sd, dc_sd

    @staticmethod
    def _uniform_mean_state_dicts(sds):
        """Moyenne uniforme (1/|S|) d'une liste de state_dicts."""
        if not sds:
            return None
        n = len(sds)
        keys = sds[0].keys()
        out = {}
        for name in keys:
            ref = sds[0][name]
            if not ref.is_floating_point():
                out[name] = ref
                continue
            out[name] = sum(sd[name] for sd in sds) / n
        return out

    @staticmethod
    def _weighted_mean_state_dicts(sds, weights):
        """Moyenne ponderee d'une liste de state_dicts par poids client."""
        if not sds:
            return None
        weights = [max(0.0, float(w)) for w in weights]
        total = sum(weights)
        if total <= 0.0:
            return ScaffoldStrategy._uniform_mean_state_dicts(sds)
        keys = sds[0].keys()
        out = {}
        for name in keys:
            ref = sds[0][name]
            if not ref.is_floating_point():
                out[name] = ref
                continue
            acc = None
            for sd, weight in zip(sds, weights):
                term = sd[name].to(ref.device) * (weight / total)
                acc = term if acc is None else acc + term
            out[name] = acc
        return out

    def _mean_state_dicts(self, sds, weights):
        if self.weighted_aggregation:
            return self._weighted_mean_state_dicts(sds, weights)
        return self._uniform_mean_state_dicts(sds)

    def aggregate_train(self, server_round, replies):
        """Agregation SCAFFOLD : moyenne des y et des delta_c + update de c_global."""
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        self._refresh_direct_downlink_bytes(contents)
        raw = sum(self._content_bytes(c) for c in self._uploaded_contents(contents))
        self.last_uplink_bytes = int(raw * self.uplink_size_ratio)
        self.last_technical_uplink_bytes = self.last_uplink_bytes
        self.last_lan_uplink_bytes = 0

        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._w_global_arrays, metrics

        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._w_global_arrays, metrics

        # 1) Decode chaque content en (y_sd, dc_sd).
        y_sds, dc_sds = [], []
        y_weights, dc_weights = [], []
        for c in accepted:
            y_sd, dc_sd = self._split_y_and_dc(c)
            mr = next(iter(c.metric_records.values()))
            weight = float(mr.get(self.weighted_by_key, 1.0))
            if weight <= 0.0:
                continue
            if y_sd:
                y_sds.append(y_sd)
                y_weights.append(weight)
            if dc_sd:
                dc_sds.append(dc_sd)
                dc_weights.append(weight)

        if not y_sds:
            return self._w_global_arrays, metrics

        # 2) w_new = w_global + eta_g * moyenne des (y_i - w_global).
        eta_g = self.server_lr
        w_global_sd = self._w_global_arrays.to_torch_state_dict()
        delta_y_per_client = [
            {k: y_sd[k] - w_global_sd[k].to(y_sd[k].device) for k in y_sd}
            for y_sd in y_sds
        ]
        delta_y_uniform = self._mean_state_dicts(delta_y_per_client, y_weights)
        new_w_sd = {}
        for k, w_g in w_global_sd.items():
            if delta_y_uniform is not None and k in delta_y_uniform:
                new_w_sd[k] = w_g + eta_g * delta_y_uniform[k].to(w_g.device)
            else:
                # Buffers non-float (ex: num_batches_tracked) : on garde le global.
                new_w_sd[k] = w_g

        # 3) Moyenne des delta_c.
        delta_c_uniform = self._mean_state_dicts(dc_sds, dc_weights)

        # 4) c_new = c_old + (|S|/N) * mean(delta_c).
        if delta_c_uniform is not None:
            n_active = len(y_sds)
            n_total = max(self.num_clients_total or n_active, 1)
            scale = n_active / n_total
            for name in self._c_global_sd:
                if name in delta_c_uniform:
                    self._c_global_sd[name] = (
                        self._c_global_sd[name]
                        + scale * delta_c_uniform[name].to(
                            self._c_global_sd[name].device)
                    )

        # 5) Diagnostics : normes de c_global et du delta_c agrege.
        cg_norm_sq = 0.0
        for v in self._c_global_sd.values():
            cg_norm_sq += float((v * v).sum().item())
        self.last_c_global_norm = float(cg_norm_sq ** 0.5)

        if delta_c_uniform is not None:
            dc_norm_sq = 0.0
            for v in delta_c_uniform.values():
                dc_norm_sq += float((v * v).sum().item())
            self.last_delta_c_norm = float(dc_norm_sq ** 0.5)
        else:
            self.last_delta_c_norm = 0.0

        return ArrayRecord(new_w_sd), metrics


class FedEveStrategy(FedAvgStrategy):
    """FedEve (Shen et al. 2025) : filtre de Kalman entre le momentum serveur
    (prediction) et la moyenne des updates clients (observation).

    Chaque round :
      1. Delta_bar = moyenne ponderee des deltas clients
      2. estime sigma2_Q (ecart momentum/observation) et sigma2_R (variance
         inter-clients)
      3. gain G_kal = sigma2_pred / (sigma2_pred + sigma2_R)
      4. M_{t+1} = M_t + G_kal * (Delta_bar - M_t)
      5. w_{t+1} = w_t - eta_g * M_{t+1}
    """

    def __init__(self, *args, server_lr=1.0, kalman_gain_min=0.05,
                 cold_start_bootstrap=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_g = float(server_lr)
        # Deux heuristiques hors papier pour stabiliser en non-IID extreme :
        # - kalman_gain_min : plancher sur G_kal (0.0 = comportement strict).
        # - cold_start_bootstrap : force G_kal=1.0 au 1er round pour amorcer
        #   le momentum (sinon M reste bloque a ~0).
        self.kalman_gain_min = float(kalman_gain_min)
        self.cold_start_bootstrap = bool(cold_start_bootstrap)
        # Etat persistant cote serveur
        self._momentum_sd = None       # M_t (init zeros au 1er configure_train)
        self._sigma2_fused = 0.0       # sigma2_t
        self._w_global_sd = None       # snapshot de w_t pour calculer les deltas
        self._bootstrapped = False
        # Diagnostics exposes pour les logs
        self.last_kalman_gain = 0.0
        self.last_sigma_Q = 0.0
        self.last_sigma_R = 0.0

    def _init_momentum(self, w_state_dict):
        """M_0 = 0 (meme shape que les params float du modele)."""
        self._momentum_sd = {
            name: torch.zeros_like(t)
            for name, t in w_state_dict.items()
            if t.is_floating_point()
        }

    def configure_train(self, server_round, arrays, config, grid):
        """Memorise w_t pour le calcul des deltas, puis broadcast w_t standard."""
        w_sd = arrays.to_torch_state_dict()
        if self._momentum_sd is None:
            self._init_momentum(w_sd)
        self._w_global_sd = {name: t.detach().clone() for name, t in w_sd.items()}
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        """Filtre de Kalman applique aux deltas clients pour produire w_{t+1}."""
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
        if not accepted:
            return self._last_global_arrays, metrics
        if self._w_global_sd is None or self._momentum_sd is None:
            # Etat non initialise -> retombe sur FedAvg.
            return aggregate_arrayrecords(accepted, self.weighted_by_key), metrics

        # 1) Extrait (local_sd, n_k) de chaque client valide.
        rows = []
        for c in accepted:
            ar = next(iter(c.array_records.values()))
            mr = next(iter(c.metric_records.values()))
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            if n_k > 0:
                rows.append((ar.to_torch_state_dict(), n_k))
        if not rows:
            return self._last_global_arrays, metrics

        # 2) Delta_k = w_t - w_local_k par client + Delta_bar pondere par n_k.
        total_n = sum(n for _, n in rows)
        client_deltas_per_param = {}  # name -> list of (delta_tensor, n_k)
        delta_avg = {}                # name -> tensor (Delta_bar)
        for name in self._momentum_sd:
            w_t = self._w_global_sd[name]
            deltas_k = []
            acc = None
            for sd, n_k in rows:
                d = w_t - sd[name].to(w_t.device)
                deltas_k.append((d, n_k))
                term = d * (n_k / total_n)
                acc = term if acc is None else acc + term
            client_deltas_per_param[name] = deltas_k
            delta_avg[name] = acc

        # 3) Estimation des variances.
        S = len(rows)
        d_total = 0
        sigma_Q_num = 0.0
        sigma_R_num = 0.0
        for name in self._momentum_sd:
            M = self._momentum_sd[name].to(delta_avg[name].device)
            d_avg = delta_avg[name]
            d_total += int(d_avg.numel())
            diff_Q = M - d_avg
            sigma_Q_num += float((diff_Q * diff_Q).sum().item())
            for d_k, _ in client_deltas_per_param[name]:
                diff_R = d_k - d_avg
                sigma_R_num += float((diff_R * diff_R).sum().item())

        if d_total > 0 and S > 0:
            sigma_Q2 = sigma_Q_num / (S * d_total)
            sigma_R2 = sigma_R_num / (S * S * d_total)
        else:
            sigma_Q2 = 0.0
            sigma_R2 = 0.0

        # 4) Variance predite et gain de Kalman.
        sigma2_pred = self._sigma2_fused + sigma_Q2
        denom = sigma2_pred + sigma_R2

        # Bootstrap au 1er round : sans lui, G_kal ~= 0 et le momentum
        # resterait bloque a 0 en non-IID extreme.
        if (not self._bootstrapped) and self.cold_start_bootstrap and S >= 2:
            G_kal = 1.0
            self._bootstrapped = True
        elif denom <= 1e-12 or S < 2:
            # Cas degenere : fallback neutre.
            G_kal = 0.5
            self._bootstrapped = True
        else:
            G_kal = sigma2_pred / denom
            self._bootstrapped = True
        G_kal = max(self.kalman_gain_min, min(1.0, float(G_kal)))

        # 5) Update momentum : M_{t+1} = M_t + G * (Delta_bar - M_t)
        new_momentum_sd = {}
        for name in self._momentum_sd:
            M = self._momentum_sd[name].to(delta_avg[name].device)
            new_momentum_sd[name] = M + G_kal * (delta_avg[name] - M)

        # 6) Update model : w_{t+1} = w_t - eta_g * M_{t+1}
        new_w_sd = {}
        for name, w_g in self._w_global_sd.items():
            if name in new_momentum_sd:
                new_w_sd[name] = w_g - self.eta_g * new_momentum_sd[name].to(w_g.device)
            else:
                new_w_sd[name] = w_g

        # 7) Update variance et persistance. Le plancher 1e-8 evite qu'un
        # round avec G_kal=1.0 fige sigma2 a 0 pour toujours.
        self._momentum_sd = {
            name: t.detach().clone() for name, t in new_momentum_sd.items()
        }
        self._sigma2_fused = max((1.0 - G_kal) * sigma2_pred, 1e-8)
        self.last_kalman_gain = float(G_kal)
        self.last_sigma_Q = float(sigma_Q2)
        self.last_sigma_R = float(sigma_R2)

        return ArrayRecord(new_w_sd), metrics


HFL_EDGE_PREFIX = "__hfl_edge_"


def _hfl_edge_prefix(edge_id):
    return f"{HFL_EDGE_PREFIX}{int(edge_id)}__"


def _pack_edge_models(edge_state_dicts):
    """Pack plusieurs modeles edge dans un seul ArrayRecord.

    Flower envoie un seul ArrayRecord a tous les clients ; on packe donc les
    modeles avec un prefix par edge (__hfl_edge_0__...) et le client selectionne
    le prefix correspondant a son edge_id.
    """
    combined = {}
    for edge_id, sd in edge_state_dicts.items():
        prefix = _hfl_edge_prefix(edge_id)
        for name, tensor in sd.items():
            combined[f"{prefix}{name}"] = tensor
    return combined

def _weighted_average_state_dicts(global_sd, rows):
    """Moyenne ponderee de state_dicts. rows = liste de (state_dict, weight)."""
    total_w = sum(float(w) for _, w in rows)
    if total_w <= 0:
        return global_sd

    new_sd = {}
    for name, w_global in global_sd.items():
        if not w_global.is_floating_point():
            new_sd[name] = w_global
            continue
        acc = None
        for sd, weight in rows:
            term = sd[name].to(w_global.device) * (float(weight) / total_w)
            acc = term if acc is None else acc + term
        new_sd[name] = acc
    return new_sd


class HierarchicalStrategy(FedAvgStrategy):
    """HFL avec etat edge persistant et sync edge-cloud periodique.

    Chaque round : les clients recoivent le modele de leur edge, entrainent,
    l'edge agrege ses clients, et le cloud synchronise les edges tous les
    `edge_local_steps` rounds.
    """

    def __init__(self, *args, num_edges=3, edge_cloud_ratio=1.0,
                 edge_cloud_bw_mbps=5.0, edge_cloud_rtt_s=0.5,
                 edge_cloud_jitter_s=0.0, edge_cloud_deadline_s=0.0,
                 straggler_sim=0, edge_local_steps=1, seed=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_edges = max(1, int(num_edges))
        self.edge_local_steps = max(1, int(edge_local_steps))
        # Compression de la portion WAN (edges <-> cloud). Remplace
        # comm_size_ratio sur cette portion (pas de double application).
        self.edge_cloud_ratio = float(edge_cloud_ratio)
        self.edge_cloud_bw_mbps = float(edge_cloud_bw_mbps)
        self.edge_cloud_rtt_s = float(edge_cloud_rtt_s)
        self.edge_cloud_jitter_s = float(edge_cloud_jitter_s)
        self.edge_cloud_deadline_s = float(edge_cloud_deadline_s)
        self.straggler_sim = int(straggler_sim)
        self.seed = int(seed)
        self.last_edge_loads = {}
        self.last_edge_cloud_time_s = 0.0
        self.last_edge_cloud_downlink_time_s = 0.0
        self.last_edge_cloud_uplink_time_s = 0.0
        self.last_edge_cloud_dropped = 0
        self.last_edge_cloud_energy_j = 0.0
        self.last_edge_cloud_downlink_energy_j = 0.0
        self.last_extra_round_time_s = 0.0
        self.last_cloud_sync = 0
        self._edge_state_dicts = {}

    def _init_edge_states(self, arrays):
        """Initialise les modeles edge depuis le modele cloud courant."""
        global_sd = arrays.to_torch_state_dict()
        self._edge_state_dicts = {
            edge: {name: tensor.detach().clone()
                   for name, tensor in global_sd.items()}
            for edge in range(self.num_edges)
        }

    def configure_train(self, server_round, arrays, config, grid):
        """Envoie a chaque client les modeles edge persistants (packes)."""
        logical_round = self._set_logical_round(server_round, config)
        self._last_global_arrays = arrays
        if (not self._edge_state_dicts
                or set(self._edge_state_dicts.keys()) != set(range(self.num_edges))):
            self._init_edge_states(arrays)

        edge_arrays = ArrayRecord(_pack_edge_models(self._edge_state_dicts))
        msgs = super().configure_train(server_round, edge_arrays, config, grid)
        technical_packed_downlink = self.last_technical_downlink_bytes
        # Le parent a memorise `edge_arrays` ; on garde le modele cloud pur
        # comme fallback/reference.
        self._last_global_arrays = arrays

        model_bytes = self._base_model_bytes(arrays)
        self.last_lan_downlink_bytes = int(
            model_bytes * len(msgs) * self.comm_size_ratio)
        self.last_technical_downlink_bytes = technical_packed_downlink
        # WAN downlink cloud->edges uniquement au demarrage et juste apres
        # une sync cloud.
        cloud_broadcast = (
            int(logical_round) == 1
            or ((int(logical_round) - 1) % self.edge_local_steps) == 0
        )
        self.last_edge_cloud_downlink_time_s = 0.0
        self.last_edge_cloud_downlink_energy_j = 0.0
        self.last_downlink_bytes = int(
            model_bytes * self.num_edges * self.edge_cloud_ratio
            if cloud_broadcast else 0)
        if cloud_broadcast and self.edge_cloud_bw_mbps > 0:
            model_mb = (model_bytes / (1024.0 * 1024.0)) * self.edge_cloud_ratio
            self.last_edge_cloud_downlink_time_s = (
                mbps_transfer_time_s(model_mb, self.edge_cloud_bw_mbps)
                + self.edge_cloud_rtt_s
            )
            self.last_edge_cloud_downlink_energy_j = compute_edge_cloud_energy_j(
                self.last_edge_cloud_downlink_time_s,
                n_links=self.num_edges,
            )
        return msgs

    def _aggregate_edge_clients(self, edge_id, edge_ref_sd, rows):
        """Hook : agrege les clients d'un edge en un modele edge.

        Par defaut moyenne ponderee FedAvg. Les sous-classes peuvent
        surcharger (ex: logique CFL dans hfl_befl_cfl/strategy.py).
        `rows` contient au minimum des tuples (sd, weight).
        """
        simple_rows = [(item[0], item[1]) for item in rows]
        return _weighted_average_state_dicts(edge_ref_sd, simple_rows)

    def _cloud_aggregate(self, global_sd, accepted_edges, logical_round):
        """Hook : agrege les modeles edges en un modele cloud.

        Par defaut moyenne ponderee FedAvg. Les sous-classes peuvent
        surcharger pour une autre agregation cloud (ex: Kalman).
        """
        return _weighted_average_state_dicts(global_sd, accepted_edges)

    def _hfl_client_groups(self, logical_round, contents, global_sd):
        """Construit edge_id -> [(state_dict, weight)] pour HFL standard.

        Hook surchargeable ; HFL-BEFL l'utilise pour appliquer FedStrag.
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
            groups.setdefault(edge_id, []).append((ar.to_torch_state_dict(), n_i))
        return groups

    def aggregate_train(self, server_round, replies):
        logical_round = self._round(server_round)
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        # LAN = clients <-> edges, WAN = edges <-> cloud.
        non_dropped = [c for c in contents if not _is_dropped(c)]
        lan_downlink_model_bytes = self._base_model_bytes(self._last_global_arrays)
        self.last_lan_downlink_bytes = int(
            lan_downlink_model_bytes * self._download_count(contents)
            * self.comm_size_ratio)
        self.last_technical_downlink_bytes = int(
            lan_downlink_model_bytes * self.num_edges
            * self._download_count(contents)
            * self.comm_size_ratio)
        raw_lan = sum(self._content_bytes(c) for c in non_dropped)
        self.last_lan_uplink_bytes = int(raw_lan * self.uplink_size_ratio)
        self.last_technical_uplink_bytes = self.last_lan_uplink_bytes
        # Le WAN uplink sera fixe plus bas, une fois connus les edges acceptes.
        self.last_uplink_bytes = 0
        downlink_time = float(getattr(self, "last_edge_cloud_downlink_time_s", 0.0))
        downlink_energy = float(
            getattr(self, "last_edge_cloud_downlink_energy_j", 0.0))
        self.last_edge_cloud_time_s = downlink_time
        self.last_edge_cloud_uplink_time_s = 0.0
        self.last_edge_cloud_dropped = 0
        self.last_edge_cloud_energy_j = downlink_energy
        self.last_extra_round_time_s = downlink_time
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if self._last_global_arrays is None:
            return None, metrics

        global_sd = self._last_global_arrays.to_torch_state_dict()
        groups = self._hfl_client_groups(logical_round, contents, global_sd)

        self.last_edge_loads = {edge: len(rows) for edge, rows in groups.items()}
        # Complete les edges sans client ce round (load=0) pour des logs complets.
        for edge in range(self.num_edges):
            self.last_edge_loads.setdefault(edge, 0)
        if not groups:
            return self._last_global_arrays, metrics

        edge_rows = []
        for edge_id, rows in groups.items():
            # Poids edge = somme des n_k de ses clients.
            edge_n = sum(item[1] for item in rows)
            edge_ref = self._edge_state_dicts.get(edge_id, global_sd)
            edge_sd = self._aggregate_edge_clients(edge_id, edge_ref, rows)
            self._edge_state_dicts[edge_id] = {
                name: tensor.detach().clone() for name, tensor in edge_sd.items()
            }
            edge_rows.append((edge_id, edge_sd, edge_n))

        do_cloud_sync = (logical_round % self.edge_local_steps) == 0
        self.last_cloud_sync = int(do_cloud_sync)
        if not do_cloud_sync:
            return self._last_global_arrays, metrics

        # Simulation du lien edge -> cloud : un modele agrege par edge actif.
        model_bytes = self._base_model_bytes(self._last_global_arrays)
        model_mb = (model_bytes / (1024.0 * 1024.0)) * self.edge_cloud_ratio
        accepted_edges = []
        if self.edge_cloud_bw_mbps > 0:
            waits = []
            for edge_id, edge_sd, edge_n in edge_rows:
                seed_base = self.seed if int(self.seed) >= 0 else 4096
                rng = random.Random(
                    seed_base + int(logical_round) * 1009 + int(edge_id) * 9176)
                jitter = 0.0
                if self.straggler_sim:
                    jitter = rng.uniform(0.0, max(0.0, self.edge_cloud_jitter_s))
                delay = (mbps_transfer_time_s(model_mb, self.edge_cloud_bw_mbps)
                         + self.edge_cloud_rtt_s + jitter)
                if (self.straggler_sim and self.edge_cloud_deadline_s > 0
                        and delay > self.edge_cloud_deadline_s):
                    self.last_edge_cloud_dropped += 1
                    waits.append(self.edge_cloud_deadline_s)
                    self.last_edge_cloud_energy_j += compute_edge_cloud_energy_j(
                        self.edge_cloud_deadline_s)
                    continue
                waits.append(delay)
                self.last_edge_cloud_energy_j += compute_edge_cloud_energy_j(delay)
                accepted_edges.append((edge_sd, edge_n))
            # Les edges uploadent en parallele : le round attend le plus lent.
            self.last_edge_cloud_uplink_time_s = max(waits, default=0.0)
            self.last_edge_cloud_time_s = (
                downlink_time + self.last_edge_cloud_uplink_time_s)
            self.last_extra_round_time_s = self.last_edge_cloud_time_s
        else:
            accepted_edges = [(edge_sd, edge_n) for _, edge_sd, edge_n in edge_rows]

        # WAN uplink final : edges -> cloud, seulement les edges acceptes.
        self.last_uplink_bytes = int(
            model_bytes * len(accepted_edges) * self.edge_cloud_ratio)

        if not accepted_edges:
            return self._last_global_arrays, metrics

        cloud_sd = self._cloud_aggregate(global_sd, accepted_edges, logical_round)
        # Apres une sync cloud, tous les edges repartent du modele cloud agrege.
        self._edge_state_dicts = {
            edge: {name: tensor.detach().clone()
                   for name, tensor in cloud_sd.items()}
            for edge in range(self.num_edges)
        }
        return ArrayRecord(cloud_sd), metrics
