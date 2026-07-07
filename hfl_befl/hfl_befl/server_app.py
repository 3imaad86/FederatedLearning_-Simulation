"""ServerApp HFL-BEFL.

Deux logiques separees, toutes les deux cote serveur :
  1) Edge assignment (RL) : _choose_edges() decide quel edge pour chaque
     client -> config["rl-edge-map"].
  2) Selection BEFL : BEFLEdgeState decide quels clients participent et
     combien d'epochs (queue Lyapunov + budget energie)
     -> config["befl-tau-per-pid"].

Le client lit ces deux listes et execute. Aucune decision cote client.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import HierarchicalStrategy

from .befl_logic import BEFLEdgeState

app = ServerApp()


class RLHierarchicalStrategy(HierarchicalStrategy):
    """HFL + affectation client-edge apprise par un petit MLP cote cloud.

    Policy-gradient REINFORCE-bandit :
       loss = - log pi(a|s) * (reward - baseline)
    avec baseline = EMA des recompenses et exploration epsilon-greedy.
    La selection BEFL (qui participe + combien d'epochs) est dans befl_state.
    """

    def __init__(self, *args, num_clients=10, rl_enabled=0,
                 rl_epsilon=0.2, rl_lr=0.01,
                 total_rounds=30, befl_V=100.0, befl_base_epochs=2,
                 befl_death_threshold=0.05, seed=-1,
                 fedstrag_enabled=0, fedstrag_alpha=1.0,
                 fedstrag_min_weight=0.05, fedstrag_max_staleness=0,
                 es_horizon_trigger=3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients = int(num_clients)
        self.rl_enabled = int(rl_enabled)
        self.rl_epsilon = float(rl_epsilon)
        self.seed = int(seed)
        self.es_horizon_trigger = max(1, int(es_horizon_trigger))

        # FedStrag : les clients late ne sont pas dropes, ils sont mis en
        # buffer puis agreges plus tard avec un poids reduit selon leur staleness.
        self.fedstrag_enabled = int(fedstrag_enabled)
        self.fedstrag_alpha = float(fedstrag_alpha)
        self.fedstrag_min_weight = float(fedstrag_min_weight)
        self.fedstrag_max_staleness = int(fedstrag_max_staleness)
        self._fedstrag_buffer = []
        self._fedstrag_due_for_round = None
        self._fedstrag_buffered_content_ids = set()
        self._fedstrag_actual_edge_loads = {}
        self.last_fedstrag_buffered = 0
        self.last_fedstrag_applied = 0
        self.last_fedstrag_dropped_stale = 0
        self.last_fedstrag_applied_pids = []
        self.accepts_late_stragglers = bool(self.fedstrag_enabled)
        # Reproductibilite RL : generator dedie pour le sampling d'actions,
        # et seeding temporaire de torch pour l'init des poids de la policy
        # (etat global sauvegarde/restaure pour ne pas perturber Flower).
        self._rl_rng = torch.Generator()
        if self.seed >= 0:
            _saved_torch_state = torch.get_rng_state()
            try:
                torch.manual_seed(self.seed * 7919 + 17)
                self.policy = nn.Sequential(
                    nn.Linear(7, 16),
                    nn.ReLU(),
                    nn.Linear(16, self.num_edges),
                )
            finally:
                torch.set_rng_state(_saved_torch_state)
            self._rl_rng.manual_seed(self.seed * 31337 + 1)
        else:
            self.policy = nn.Sequential(
                nn.Linear(7, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_edges),
            )
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=float(rl_lr))
        self.last_client_stats = {}
        self.last_edge_map = [i % self.num_edges for i in range(self.num_clients)]

        # Etat REINFORCE : baseline = EMA des recompenses par round.
        self._reward_baseline = 0.0
        self._baseline_alpha = 0.1
        self._baseline_initialized = False
        # Flags des replies stale FedStrag, a exclure de l'apprentissage RL.
        self._last_learn_stale_flags = None

        # Etat BEFL (selection clients + tau via Lyapunov).
        self.befl_state = BEFLEdgeState(
            num_clients=self.num_clients,
            total_rounds=total_rounds,
            V=befl_V,
            base_epochs_default=befl_base_epochs,
            death_threshold=befl_death_threshold,
            weighted_by_key=str(self.weighted_by_key),
        )

        # Diagnostics agreges batterie/queue, calcules apres chaque round.
        self.last_battery_min_ratio = 1.0
        self.last_battery_mean_ratio = 1.0
        self.last_battery_var_ratio = 0.0
        self.last_queue_mean = 0.0
        self.last_queue_max = 0.0

    def _fedstrag_weight_from_staleness(self, stale):
        """Poids FedStrag = 1 / (1 + alpha * staleness), borne par min_weight."""
        if not self.fedstrag_enabled:
            return 1.0
        stale = max(0.0, float(stale))
        if stale <= 0.0:
            return 1.0
        if self.fedstrag_max_staleness > 0 and stale > self.fedstrag_max_staleness:
            return 0.0
        alpha = max(0.0, self.fedstrag_alpha)
        scale = 1.0 / (1.0 + alpha * stale)
        floor = max(0.0, min(1.0, self.fedstrag_min_weight))
        return max(floor, min(1.0, scale))

    @staticmethod
    def _fedstrag_is_late(metric_record):
        return float(metric_record.get("fedstrag_late", 0.0)) >= 0.5

    @staticmethod
    def _metric_record(content):
        return next(iter(content.metric_records.values()))

    @staticmethod
    def _metric_snapshot(metric_record):
        return {k: v for k, v in metric_record.items()}

    @staticmethod
    def _clone_state_dict(state_dict):
        return {
            name: tensor.detach().clone()
            for name, tensor in state_dict.items()
        }

    @staticmethod
    def _rebased_stale_state(current_sd, base_sd, stale_sd):
        """Applique le delta stale sur l'etat edge courant : current + (stale - base)."""
        rebased = {}
        for name, current in current_sd.items():
            if not current.is_floating_point():
                rebased[name] = current
                continue
            if name not in stale_sd or name not in base_sd:
                rebased[name] = current
                continue
            delta = stale_sd[name].to(current.device) - base_sd[name].to(current.device)
            rebased[name] = current + delta
        return rebased

    def _edge_state_for_buffer(self, edge_id):
        base = self._edge_state_dicts.get(int(edge_id))
        if base is not None:
            return base
        if self._last_global_arrays is not None:
            return self._last_global_arrays.to_torch_state_dict()
        return None

    def _pop_fedstrag_due(self, logical_round):
        """Retourne les updates stale qui arrivent au round courant."""
        if not self._fedstrag_buffer:
            return []
        due, pending = [], []
        for item in self._fedstrag_buffer:
            if int(item["arrival_round"]) <= int(logical_round):
                due.append(item)
            else:
                pending.append(item)
        self._fedstrag_buffer = pending
        return due

    def _buffer_fedstrag_update(self, content, logical_round, edge_id, base_sd=None):
        """Met une update late en file d'attente et retourne True si consommee."""
        if not self.fedstrag_enabled:
            return False
        ar = next(iter(content.array_records.values()))
        mr = self._metric_record(content)
        if not self._fedstrag_is_late(mr):
            return False
        n_i = float(mr.get(self.weighted_by_key, 0.0))
        if n_i <= 0.0:
            self.last_fedstrag_dropped_stale += 1
            return True
        if base_sd is None:
            base_sd = self._edge_state_for_buffer(edge_id)
        if base_sd is None:
            self.last_fedstrag_dropped_stale += 1
            return True
        stale = max(1, int(float(mr.get("fedstrag_staleness", 1.0))))
        # Borne dure sur le buffer pour eviter une fuite memoire si un
        # client reste durablement en panne reseau.
        max_buffer = max(8, 2 * int(self.num_clients))
        if len(self._fedstrag_buffer) >= max_buffer:
            self.last_fedstrag_dropped_stale += 1
            return True
        pid = int(mr.get("partition_id", -1))
        self._fedstrag_buffer.append({
            "arrival_round": int(logical_round) + stale,
            "edge_id": int(edge_id),
            "base_state_dict": self._clone_state_dict(base_sd),
            "state_dict": self._clone_state_dict(ar.to_torch_state_dict()),
            "metrics": self._metric_snapshot(mr),
            "features": (
                self._features(pid).detach().clone() if pid >= 0 else None
            ),
            "pid": pid,
            "num_examples": n_i,
            "staleness": stale,
            # Marqueur off-policy : a exclure de l'apprentissage RL.
            "is_stale": True,
        })
        self.last_fedstrag_buffered += 1
        return True

    def _hfl_client_groups(self, logical_round, contents, global_sd):
        """Groupes edge -> clients, avec FedStrag si active.

        - replies normales : agregation edge immediate avec poids n_i.
        - replies late     : bufferisees, puis reinjectees a leur round
                             d'arrivee avec poids n_i * stale_weight.
        """
        if not self.fedstrag_enabled:
            return super()._hfl_client_groups(logical_round, contents, global_sd)

        groups = {}
        due_items = self._fedstrag_due_for_round
        if due_items is None:
            due_items = self._pop_fedstrag_due(logical_round)

        for item in due_items:
            scale = self._fedstrag_weight_from_staleness(item["staleness"])
            n_i = float(item["num_examples"])
            stale_weight = n_i * scale
            if stale_weight <= 0.0:
                self.last_fedstrag_dropped_stale += 1
                continue
            edge_id = int(item["edge_id"]) % self.num_edges
            current_sd = self._edge_state_dicts.get(edge_id, global_sd)
            rebased_sd = self._rebased_stale_state(
                current_sd, item["base_state_dict"], item["state_dict"])
            groups.setdefault(edge_id, []).append((rebased_sd, stale_weight))
            anchor_weight = n_i * max(0.0, 1.0 - scale)
            if anchor_weight > 0.0:
                groups.setdefault(edge_id, []).append((current_sd, anchor_weight))
            self.last_fedstrag_applied += 1
            pid = int(item.get("pid", -1))
            if pid >= 0:
                self.last_fedstrag_applied_pids.append(pid)
            self._fedstrag_actual_edge_loads[edge_id] = (
                self._fedstrag_actual_edge_loads.get(edge_id, 0) + 1)

        for content in contents:
            try:
                mr = self._metric_record(content)
                ar = next(iter(content.array_records.values()))
            except (ValueError, StopIteration, AttributeError):
                continue
            if float(mr.get("dropped", 0.0)) >= 0.5:
                continue
            edge_id = int(mr.get("hfl_edge_id", 0)) % self.num_edges
            if self._fedstrag_is_late(mr):
                if id(content) not in self._fedstrag_buffered_content_ids:
                    self._buffer_fedstrag_update(
                        content, logical_round, edge_id,
                        self._edge_state_for_buffer(edge_id))
                    self._fedstrag_buffered_content_ids.add(id(content))
                continue
            n_i = float(mr.get(self.weighted_by_key, 0.0))
            if n_i <= 0.0:
                continue
            groups.setdefault(edge_id, []).append((ar.to_torch_state_dict(), n_i))
            self._fedstrag_actual_edge_loads[edge_id] = (
                self._fedstrag_actual_edge_loads.get(edge_id, 0) + 1)

        return groups

    def _features(self, pid):
        """Etat observable du client pour le RL (7 features, normalisees ~[0,1])."""
        import math
        stats = self.last_client_stats.get(int(pid), {})
        tier = float(stats.get("tier", int(pid) % 3)) / 2.0
        net = float(stats.get("net_tier", int(pid) % 3)) / 2.0
        battery = float(stats.get("battery", 1.0))
        # Echelle log : garde un gradient utile entre 10s et 5min.
        time_raw = float(stats.get("time", 0.0))
        time_s = min(math.log1p(time_raw) / math.log1p(60.0), 1.0)
        dropped = float(stats.get("dropped", 0.0))

        # Queue Lyapunov normalisee par le cout typique d'un round.
        q_raw = float(self.befl_state.queue.get(int(pid), 0.0))
        e_per_ep = float(self.befl_state.energy_per_epoch.get(int(pid), 0.0))
        scale = max(1.0, e_per_ep * float(self.befl_state.base_epochs_default))
        lyapunov_q = min(q_raw / scale, 5.0) / 5.0

        # Variance des charges edge au dernier round.
        loads = list(self.last_edge_loads.values())
        if len(loads) >= 2:
            mean_l = sum(loads) / len(loads)
            var_l = sum((l - mean_l) ** 2 for l in loads) / len(loads)
            edge_var = min(var_l / max(1.0, (self.num_clients / 2.0) ** 2), 1.0)
        else:
            edge_var = 0.0

        return torch.tensor(
            [tier, net, battery, time_s, dropped, lyapunov_q, edge_var],
            dtype=torch.float32,
        )

    def _mixed_policy_probs(self, logits):
        """Policy epsilon-soft: (1-eps)*pi_theta + eps*Uniform."""
        probs = F.softmax(logits, dim=-1)
        if self.rl_epsilon <= 0:
            return probs
        uniform = torch.full_like(probs, 1.0 / float(self.num_edges))
        return (1.0 - self.rl_epsilon) * probs + self.rl_epsilon * uniform

    def _choose_edges(self):
        """Echantillonne un edge par client dans la distribution epsilon-soft."""
        edge_map = []
        for pid in range(self.num_clients):
            with torch.no_grad():
                logits = self.policy(self._features(pid))
                probs = self._mixed_policy_probs(logits)
                edge = int(torch.multinomial(
                    probs, 1, generator=self._rl_rng).item())
            edge_map.append(edge)
        return edge_map

    def configure_train(self, server_round, arrays, config, grid):
        logical_round = self._set_logical_round(server_round, config)
        # 1) BEFL : decide tau_i pour chaque pid (-1 = base_epochs, 0 = drop).
        taus = self.befl_state.decide_taus(logical_round)
        self._set_config_or_raise(
            config, "befl-tau-per-pid", [int(t) for t in taus])

        # 2) RL : decide edge_id pour chaque pid.
        if self.rl_enabled:
            self.last_edge_map = self._choose_edges()
            self._set_config_or_raise(
                config, "rl-edge-map", [int(e) for e in self.last_edge_map])

        # 3) Genere les messages habituels (un par client connecte).
        return super().configure_train(server_round, arrays, config, grid)

    @staticmethod
    def _set_config_or_raise(config, key, value):
        """Ecrit une decision serveur dans le train_config sans fallback muet."""
        try:
            config[key] = value
        except Exception as exc:
            raise RuntimeError(
                f"Impossible d'injecter {key!r} dans train_config; "
                "les clients HFL-BEFL recevraient un fallback silencieux."
            ) from exc
        if config.get(key, None) is None:
            raise RuntimeError(
                f"train_config n'a pas conserve {key!r}; "
                "les clients HFL-BEFL recevraient un fallback silencieux."
            )

    @staticmethod
    def _reward(mr):
        dropped = float(mr.get("dropped", 0.0))
        time_s = float(mr.get("local_time_s", 0.0))
        comm_s = float(mr.get("comm_time_s", 0.0))
        energy = float(mr.get("energy_j", 0.0))
        return 1.0 - 2.0 * dropped - 0.01 * (time_s + comm_s) - 0.001 * energy

    def _learn_from_metric_records(self, metric_records, feature_records=None):
        """Mise a jour REINFORCE-bandit de la policy :
            loss = - log pi(a|s) * (reward - baseline), grad clip 1.0.
        """
        if not self.rl_enabled:
            return
        losses = []
        rewards_collected = []
        feature_records = feature_records or [None] * len(metric_records)
        # Skip les replies stale FedStrag : theta a change depuis le choix
        # de l'action, l'apprentissage serait biaise (off-policy).
        stale_flags = self._last_learn_stale_flags or [False] * len(metric_records)
        for mr, feature_snapshot, is_stale in zip(
                metric_records, feature_records, stale_flags):
            if mr is None:
                continue
            if is_stale:
                continue
            pid = int(mr.get("partition_id", -1))
            edge = int(mr.get("hfl_edge_id", 0)) % self.num_edges
            if pid < 0:
                continue
            # Si le drop est decide par BEFL (cote serveur), l'edge choisi
            # n'a pas influence le resultat -> ne pas penaliser la policy.
            if float(mr.get("befl_drop", 0.0)) >= 0.5:
                continue
            reward = self._reward(mr)
            rewards_collected.append(float(reward))
            features = feature_snapshot if feature_snapshot is not None else self._features(pid)
            logits = self.policy(features)
            probs = self._mixed_policy_probs(logits)
            advantage = float(reward) - self._reward_baseline
            action_prob = torch.clamp(probs[edge], min=1e-8)
            losses.append(-torch.log(action_prob) * advantage)
        if losses:
            loss = torch.stack(losses).mean()
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optim.step()
        # MAJ baseline (apres backward pour ne pas la melanger au gradient).
        if rewards_collected:
            mean_r = sum(rewards_collected) / len(rewards_collected)
            if not self._baseline_initialized:
                self._reward_baseline = mean_r
                self._baseline_initialized = True
            else:
                self._reward_baseline = (
                    (1.0 - self._baseline_alpha) * self._reward_baseline
                    + self._baseline_alpha * mean_r
                )

    def _learn_from_replies(self, contents):
        metric_records = []
        for content in contents:
            if content is None:
                continue
            try:
                metric_records.append(self._metric_record(content))
            except (StopIteration, AttributeError):
                continue
        self._learn_from_metric_records(metric_records)

    def _update_client_stats_from_metric_records(self, metric_records):
        """Met a jour les features RL pour le prochain round.

        La batterie restante est suivie cote serveur par BEFLEdgeState (le
        client renvoie -1 comme sentinelle).
        """
        for mr in metric_records:
            if mr is None:
                continue
            pid = int(mr.get("partition_id", -1))
            if pid < 0:
                continue

            cap = float(self.befl_state.battery.get(
                pid, float(mr.get("battery_capacity_j", 0.0))))
            if cap <= 0:
                battery_ratio = 1.0
            else:
                used = float(self.befl_state.energy_used.get(pid, 0.0))
                battery_ratio = max(0.0, min(1.0, (cap - used) / cap))

            self.last_client_stats[pid] = {
                "tier": float(mr.get("resource_tier", int(pid) % 3)),
                "net_tier": float(mr.get("net_tier", int(pid) % 3)),
                "battery": battery_ratio,
                "time": float(mr.get("local_time_s", 0.0)),
                "dropped": float(mr.get("dropped", 0.0)),
            }

    def _update_client_stats_from_replies(self, replies):
        metric_records = []
        for msg in replies:
            try:
                metric_records.append(next(iter(msg.content.metric_records.values())))
            except (ValueError, StopIteration, AttributeError):
                continue
        self._update_client_stats_from_metric_records(metric_records)

    def notify_es_progress(self, no_improve, patience):
        """Raccourcit l'horizon Lyapunov quand l'early-stopping approche,
        pour que les clients osent depenser leur batterie sur la fin."""
        if patience <= 0:
            self.befl_state.set_remaining_rounds_hint(None)
            return
        rounds_until_es = max(1, patience - int(no_improve))
        if rounds_until_es <= int(self.es_horizon_trigger):
            self.befl_state.set_remaining_rounds_hint(rounds_until_es)
        else:
            self.befl_state.set_remaining_rounds_hint(None)

    def aggregate_train(self, server_round, replies):
        logical_round = self._round(server_round)
        replies = list(replies)

        # Filtre les Messages sans content (client crashe, OOM, deadline).
        valid_replies = []
        for msg in replies:
            try:
                if getattr(msg, "has_content", lambda: True)() and msg.content is not None:
                    valid_replies.append(msg)
            except Exception:
                continue

        self.last_fedstrag_buffered = 0
        self.last_fedstrag_applied = 0
        self.last_fedstrag_dropped_stale = 0
        self.last_fedstrag_applied_pids = []
        self._fedstrag_buffered_content_ids = set()
        self._fedstrag_actual_edge_loads = {}

        due_items = []
        if self.fedstrag_enabled:
            due_items = self._pop_fedstrag_due(logical_round)
            self._fedstrag_due_for_round = due_items

        # Les replies late ne participent pas a l'agregation avant leur round
        # d'arrivee, mais leur energie a deja ete consommee : BEFL met donc a
        # jour batterie/queue immediatement.
        immediate_metric_records = []
        befl_metric_records = []
        for msg in valid_replies:
            try:
                content = msg.content
                mr = self._metric_record(content)
            except (ValueError, StopIteration, AttributeError):
                continue
            befl_metric_records.append(mr)
            late = (
                self.fedstrag_enabled
                and float(mr.get("dropped", 0.0)) < 0.5
                and self._fedstrag_is_late(mr)
            )
            if late:
                edge_id = int(mr.get("hfl_edge_id", 0)) % self.num_edges
                self._buffer_fedstrag_update(
                    content, logical_round, edge_id,
                    self._edge_state_for_buffer(edge_id))
                self._fedstrag_buffered_content_ids.add(id(content))
                continue
            immediate_metric_records.append(mr)

        due_metric_records = [item.get("metrics") for item in due_items]
        due_feature_records = [item.get("features") for item in due_items]
        state_metric_records = immediate_metric_records + due_metric_records
        state_feature_records = (
            [None] * len(immediate_metric_records) + due_feature_records
        )
        self._last_learn_stale_flags = (
            [False] * len(immediate_metric_records)
            + [True] * len(due_metric_records)
        )

        # Ordre important : learn AVANT update, pour que _features lise la
        # meme queue (celle du round r-1) au moment du choix de l'action et
        # au moment du calcul de log_pi.
        self._learn_from_metric_records(state_metric_records, state_feature_records)

        # BEFL : maj de Q + energie + num_examples pour chaque client.
        self.befl_state.update_from_metric_records(befl_metric_records, logical_round)
        self._update_client_stats_from_metric_records(state_metric_records)

        # Carte pid -> batterie restante que le runner injecte dans les logs.
        self.last_battery_remaining_j = {}
        for pid, cap in self.befl_state.battery.items():
            if cap > 0:
                used = self.befl_state.energy_used.get(pid, 0.0)
                self.last_battery_remaining_j[int(pid)] = max(0.0, cap - used)

        # Stats agregees batterie et queue Lyapunov.
        ratios = []
        for pid, cap in self.befl_state.battery.items():
            if cap > 0:
                used = self.befl_state.energy_used.get(pid, 0.0)
                ratios.append(max(0.0, 1.0 - used / cap))
        if ratios:
            self.last_battery_min_ratio = float(min(ratios))
            self.last_battery_mean_ratio = float(sum(ratios) / len(ratios))
            mean_r = self.last_battery_mean_ratio
            var = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
            self.last_battery_var_ratio = float(var)
        else:
            self.last_battery_min_ratio = 1.0
            self.last_battery_mean_ratio = 1.0
            self.last_battery_var_ratio = 0.0
        # Queue Lyapunov : uniquement sur les clients a batterie finie
        # (les unlimited ont Q = 0 et fausseraient la moyenne).
        constrained_queues = [
            float(q) for pid, q in self.befl_state.queue.items()
            if self.befl_state.battery.get(pid, 0.0) > 0
        ]
        if constrained_queues:
            self.last_queue_mean = float(
                sum(constrained_queues) / len(constrained_queues))
            self.last_queue_max = float(max(constrained_queues))
        else:
            self.last_queue_mean = 0.0
            self.last_queue_max = 0.0

        # Agregation hierarchique standard (le parent compte aussi les drops).
        try:
            result = super().aggregate_train(server_round, replies)
        finally:
            self._fedstrag_due_for_round = None
        if self.fedstrag_enabled:
            self.last_edge_loads = dict(self._fedstrag_actual_edge_loads)
        return result


def _tail(_, strategy):
    loads = ",".join(
        f"e{edge}:{count}" for edge, count in sorted(strategy.last_edge_loads.items())
    )
    edge_time = getattr(strategy, "last_edge_cloud_time_s", 0.0)
    edge_drop = getattr(strategy, "last_edge_cloud_dropped", 0)
    cloud_sync = getattr(strategy, "last_cloud_sync", 0)
    local_steps = getattr(strategy, "edge_local_steps", 1)
    rl = getattr(strategy, "rl_enabled", 0)
    baseline = getattr(strategy, "_reward_baseline", 0.0)
    rl_tail = f" rl={rl}"
    if rl:
        rl_tail += f" rl_baseline={baseline:+.3f}"
    fedstrag_tail = ""
    if getattr(strategy, "fedstrag_enabled", 0):
        fedstrag_tail = (
            f" fedstrag_buffer={len(getattr(strategy, '_fedstrag_buffer', []))}"
            f" fedstrag_buf={getattr(strategy, 'last_fedstrag_buffered', 0)}"
            f" fedstrag_apply={getattr(strategy, 'last_fedstrag_applied', 0)}"
            f" fedstrag_drop={getattr(strategy, 'last_fedstrag_dropped_stale', 0)}"
        )
    # Diagnostics BEFL, affiches uniquement si au moins un client est constrained.
    befl_tail = ""
    bat_mean = float(getattr(strategy, "last_battery_mean_ratio", 1.0))
    if bat_mean < 1.0:
        bat_min = float(getattr(strategy, "last_battery_min_ratio", 1.0))
        bat_var = float(getattr(strategy, "last_battery_var_ratio", 0.0))
        q_mean = float(getattr(strategy, "last_queue_mean", 0.0))
        q_max = float(getattr(strategy, "last_queue_max", 0.0))
        befl_tail = (
            f" bat=mean{bat_mean:.2f}/min{bat_min:.2f}/var{bat_var:.1e}"
            f" Q=mean{q_mean:.1f}/max{q_max:.1f}"
        )
    if not loads:
        return (f" edges=[] k1={local_steps} sync={cloud_sync}"
                f"{rl_tail}{befl_tail}{fedstrag_tail}")
    return (f" edges=[{loads}] k1={local_steps} sync={cloud_sync}"
            f" edge_cloud={edge_time:.2f}s"
            f" edge_drop={edge_drop}{rl_tail}{befl_tail}{fedstrag_tail}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    num_edges = int(cfg.get("hfl-num-edges", 3))
    edge_cloud_ratio = float(cfg.get("hfl-edge-cloud-ratio", 0.25))
    edge_cloud_bw = float(cfg.get("hfl-edge-cloud-bw-mbps", 5.0))
    edge_cloud_rtt = float(cfg.get("hfl-edge-cloud-rtt-s", 0.5))
    edge_cloud_jitter = float(cfg.get("hfl-edge-cloud-jitter-s", 0.0))
    edge_cloud_deadline = float(cfg.get("hfl-edge-cloud-deadline-s", 0.0))
    edge_local_steps = int(cfg.get("hfl-local-steps", 3))
    straggler_sim = int(cfg.get("straggler-sim", 0))
    unlimited_tier = int(cfg.get("befl-unlimited-tier", 2))
    num_clients = int(cfg.get("num-clients", 10))
    rl_enabled = int(cfg.get("hfl-rl-edge-assignment", 0))
    rl_epsilon = float(cfg.get("hfl-rl-epsilon", 0.2))
    rl_lr = float(cfg.get("hfl-rl-lr", 0.01))
    total_rounds = int(cfg.get("num-server-rounds", 30))
    befl_V = float(cfg.get("befl-V", 100.0))
    befl_base_epochs = int(cfg.get("local-epochs", 2))
    befl_death_threshold = float(cfg.get("befl-death-threshold", 0.05))
    fedstrag_enabled = int(cfg.get("fedstrag-enabled", 0))
    fedstrag_alpha = float(cfg.get("fedstrag-alpha", 1.0))
    fedstrag_min_weight = float(cfg.get("fedstrag-min-weight", 0.05))
    fedstrag_max_staleness = int(cfg.get("fedstrag-max-staleness", 0))
    es_horizon_trigger = int(cfg.get("befl-es-horizon-trigger", 3))
    if fedstrag_enabled and not straggler_sim:
        print("[HFL-BEFL] WARN: fedstrag-enabled=1 mais straggler-sim=0 : "
              "aucun reply late ne sera genere, le buffering FedStrag "
              "restera code mort. Active straggler-sim=1 si tu veux "
              "tester FedStrag.")
    rl_seed = int(cfg.get("seed", -1))
    if num_edges < 1:
        raise ValueError("hfl-num-edges doit etre >= 1")
    if edge_local_steps < 1:
        raise ValueError("hfl-local-steps doit etre >= 1")
    if edge_cloud_ratio < 0.0:
        raise ValueError("hfl-edge-cloud-ratio doit etre >= 0")
    if not (0.0 <= rl_epsilon <= 1.0):
        raise ValueError("hfl-rl-epsilon doit etre dans [0, 1]")
    if rl_lr <= 0.0:
        raise ValueError("hfl-rl-lr doit etre > 0")
    if befl_V < 0.0:
        raise ValueError("befl-V doit etre >= 0")
    if not (0.0 <= befl_death_threshold <= 1.0):
        raise ValueError("befl-death-threshold doit etre dans [0, 1]")
    if fedstrag_enabled not in (0, 1):
        raise ValueError("fedstrag-enabled doit etre 0 ou 1")
    if fedstrag_alpha < 0.0:
        raise ValueError("fedstrag-alpha doit etre >= 0")
    if not (0.0 <= fedstrag_min_weight <= 1.0):
        raise ValueError("fedstrag-min-weight doit etre dans [0, 1]")
    if fedstrag_max_staleness < 0:
        raise ValueError("fedstrag-max-staleness doit etre >= 0")

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="HFL-BEFL",
        strategy_class=RLHierarchicalStrategy,
        strategy_kwargs={
            "num_edges": num_edges,
            "num_clients": num_clients,
            "edge_cloud_ratio": edge_cloud_ratio,
            "edge_cloud_bw_mbps": edge_cloud_bw,
            "edge_cloud_rtt_s": edge_cloud_rtt,
            "edge_cloud_jitter_s": edge_cloud_jitter,
            "edge_cloud_deadline_s": edge_cloud_deadline,
            "edge_local_steps": edge_local_steps,
            "straggler_sim": straggler_sim,
            "rl_enabled": rl_enabled,
            "rl_epsilon": rl_epsilon,
            "rl_lr": rl_lr,
            "total_rounds": total_rounds,
            "befl_V": befl_V,
            "befl_base_epochs": befl_base_epochs,
            "befl_death_threshold": befl_death_threshold,
            "seed": rl_seed,
            "fedstrag_enabled": fedstrag_enabled,
            "fedstrag_alpha": fedstrag_alpha,
            "fedstrag_min_weight": fedstrag_min_weight,
            "fedstrag_max_staleness": fedstrag_max_staleness,
            "es_horizon_trigger": es_horizon_trigger,
        },
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="hfl_befl",
        extra_tail_fn=_tail,
        banner_extra=(
            f" edges={num_edges} edge_cloud_ratio={edge_cloud_ratio}"
            f" edge_cloud_bw={edge_cloud_bw}Mbps edge_cloud_rtt={edge_cloud_rtt}s"
            f" edge_cloud_jitter={edge_cloud_jitter}s"
            f" edge_cloud_deadline={edge_cloud_deadline}s"
            f" local_steps={edge_local_steps}"
            f" unlimited_tier={unlimited_tier}"
            f" rl_edge={rl_enabled}"
            f" fedstrag={fedstrag_enabled}"
            f" fedstrag_alpha={fedstrag_alpha}"
        ),
    )
