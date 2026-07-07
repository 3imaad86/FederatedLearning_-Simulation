"""Logique BEFL (Liu et al. 2022) -- equilibrage de l'energie.

3 etapes par round :
  1) compute_budget_per_round : energie autorisee ce round
                                = batterie_restante / rounds_restants
  2) decide_tau               : combien d'epochs tiennent dans ce budget
  3) should_select            : critere Lyapunov V*u >= Q*E*tau

Apres le round, update_queue fait evoluer la queue Q : si un client a
surconsomme recemment, Q grossit et il est ecarte jusqu'a ce que Q se vide.
"""


def compute_budget_per_round(battery_capacity_j, energy_used, rounds_remaining):
    """Budget energie du round = batterie restante / rounds restants.

    battery_capacity_j <= 0 -> mode unlimited (renvoie infini).
    """
    if battery_capacity_j <= 0:
        return float("inf")
    if rounds_remaining <= 0:
        return 0.0
    remaining_j = battery_capacity_j - energy_used
    if remaining_j < 0:
        remaining_j = 0.0
    return remaining_j / rounds_remaining


def decide_tau(base_epochs, budget_per_round, energy_per_epoch, fixed_energy=0.0):
    """Nombre d'epochs possibles dans le budget (base_epochs si energie inconnue)."""
    if energy_per_epoch <= 0:
        return int(base_epochs)
    budget_for_epochs = float(budget_per_round) - float(fixed_energy)
    if budget_for_epochs <= 0:
        return 0
    max_by_budget = int(budget_for_epochs / energy_per_epoch)
    if max_by_budget < 0:
        max_by_budget = 0
    if max_by_budget > int(base_epochs):
        return int(base_epochs)
    return max_by_budget


def should_select(queue, energy_per_epoch, tau, utility, V, fixed_energy=0.0):
    """Critere Lyapunov : participe ssi V*utilite >= Q*E*tau.

    Comparaison `>=` (pas `>` strict) : au 1er round Q=0 donc benefit ==
    cost == 0, et il faut laisser le client participer.
    """
    if tau <= 0:
        return False
    expected_energy = float(fixed_energy) + float(energy_per_epoch) * float(tau)
    cost = float(queue) * expected_energy
    benefit = float(V) * float(utility)
    return benefit >= cost


def update_queue(queue, energy_consumed, budget_per_round):
    """Mise a jour Lyapunov : Q_new = max(0, Q + E_consommee - budget)."""
    if budget_per_round == float("inf"):
        return 0.0
    new_q = float(queue) + float(energy_consumed) - float(budget_per_round)
    if new_q < 0:
        return 0.0
    return new_q


class BEFLEdgeState:
    """Etat BEFL maintenu cote serveur : decide la selection et le nombre
    d'epochs tau_i de chaque client a chaque round.

    Pour chaque client connu (apres son 1er reply), on stocke sa batterie,
    l'energie consommee, l'estimation J/epoch et sa queue Q de Lyapunov.

    Q est trackee par client (pas par edge) : la dette energetique est une
    propriete de la batterie du device, elle suit le client meme si le RL
    le reassigne a un autre edge. Une seule instance sert donc tous les edges.
    """

    def __init__(self, num_clients, total_rounds, V=100.0,
                 base_epochs_default=2, death_threshold=0.05,
                 weighted_by_key="num-examples"):
        self.num_clients = int(num_clients)
        self.total_rounds = int(total_rounds)
        self.V = float(V)
        self.base_epochs_default = int(base_epochs_default)
        self.death_threshold = float(death_threshold)
        self.weighted_by_key = str(weighted_by_key)
        # Etat par pid (rempli au fur et a mesure des replies)
        self.battery = {}            # pid -> battery_capacity_j
        self.energy_used = {}        # pid -> J cumule
        self.energy_per_epoch = {}   # pid -> J/epoch compute estime
        self.fixed_energy = {}       # pid -> J fixe/round (communication)
        self.queue = {}              # pid -> Q (Lyapunov)
        # Utility = taille de partition n_k, normalisee par la moyenne.
        self.num_examples = {}       # pid -> n_k
        # mean_n est fige une fois assez de clients observes, pour eviter
        # que l'echelle de utility ne derive sous participation partielle.
        self._mean_n_frozen = None

        # Hint pour raccourcir l'horizon quand l'early-stopping approche
        # (sinon la batterie reste sous-exploitee). None = total_rounds.
        self._remaining_rounds_hint = None

        # base_epochs naturel de chaque client (son tier), appris au 1er
        # reply utile, pour ne pas gonfler tau au-dessus du naturel.
        self.base_epochs_per_pid = {}

    def set_remaining_rounds_hint(self, n):
        """Override de l'horizon restant (None = comportement normal)."""
        self._remaining_rounds_hint = (
            None if n is None else max(1, int(n))
        )

    def _effective_rounds_remaining(self, server_round):
        """Rounds restants pour le calcul du budget (hint applique si fourni)."""
        natural = max(1, self.total_rounds - server_round + 1)
        if self._remaining_rounds_hint is None:
            return natural
        return max(1, min(natural, self._remaining_rounds_hint))

    def decide_taus(self, server_round):
        """Decide tau pour chaque pid. Renvoie une liste de num_clients entiers.

        Codes :
          -1 = client jamais vu -> qu'il fasse base_epochs
           0 = drop (Lyapunov ou batterie morte)
          >0 = nombre d'epochs a faire ce round
        """
        out = []
        rounds_remaining = self._effective_rounds_remaining(server_round)

        # Moyenne des n_k connus pour normaliser l'utility ; figee des que
        # 75% des clients ont ete observes.
        if self._mean_n_frozen is not None:
            mean_n = self._mean_n_frozen
        else:
            all_n = list(self.num_examples.values())
            mean_n = sum(all_n) / len(all_n) if all_n else 1.0
            if len(all_n) >= max(3, int(0.75 * self.num_clients)):
                self._mean_n_frozen = float(mean_n)

        for pid in range(self.num_clients):
            # 1er round / client jamais vu : on laisse passer
            if pid not in self.battery:
                out.append(-1)
                continue

            battery_j = self.battery[pid]
            # Mode unlimited : pas de BEFL, full base_epochs
            if battery_j <= 0:
                out.append(-1)
                continue

            energy_used = self.energy_used[pid]
            energy_per_epoch = self.energy_per_epoch[pid]
            fixed_energy = self.fixed_energy.get(pid, 0.0)
            queue = self.queue[pid]

            # Death check (batterie quasi vide)
            ratio = max(0.0, 1.0 - energy_used / battery_j)
            if ratio < self.death_threshold:
                out.append(0)
                continue

            # 1) budget pour ce round
            budget = compute_budget_per_round(
                battery_j, energy_used, rounds_remaining)
            # 2) combien d'epochs, cap au base_epochs naturel du client
            client_base_ep = self.base_epochs_per_pid.get(
                pid, self.base_epochs_default)
            tau = decide_tau(client_base_ep, budget, energy_per_epoch,
                             fixed_energy=fixed_energy)
            # 3) selection Lyapunov avec utility = n_k / mean_n
            n_k = self.num_examples.get(pid, mean_n)
            utility = float(n_k) / max(mean_n, 1.0)
            if not should_select(queue, energy_per_epoch, tau, utility, self.V,
                                 fixed_energy=fixed_energy):
                out.append(0)
            else:
                out.append(tau)

        return out

    def update_from_metric_records(self, metric_records, server_round):
        """Met a jour Q + energie depuis une liste de MetricRecord/dicts."""
        rounds_remaining = self._effective_rounds_remaining(server_round)

        for mr in metric_records:
            if mr is None:
                continue
            pid = int(mr.get("partition_id", -1))
            if pid < 0:
                continue

            battery_capacity = float(mr.get("battery_capacity_j", 0.0))
            energy_j = float(mr.get("energy_j", 0.0))
            compute_energy_j = float(mr.get("compute_energy_j", energy_j))
            epochs_used = float(mr.get("epochs_used", 0.0))

            # Init au 1er reply
            first_reply = pid not in self.battery
            if first_reply:
                self.battery[pid] = battery_capacity
                self.energy_used[pid] = 0.0
                self.energy_per_epoch[pid] = 0.0
                self.fixed_energy[pid] = 0.0
                self.queue[pid] = 0.0

            # Bootstrap du base_epochs naturel au 1er reply utile
            # (epochs_used > 0, donc pas un drop).
            if pid not in self.base_epochs_per_pid and epochs_used > 0:
                self.base_epochs_per_pid[pid] = int(epochs_used)

            # Mode unlimited : on ne tracke rien
            if self.battery[pid] <= 0:
                continue

            # Budget calcule AVANT d'ajouter la conso (pour le bon Q)
            old_energy_used = self.energy_used[pid]
            budget = compute_budget_per_round(
                self.battery[pid], old_energy_used, rounds_remaining)
            self.queue[pid] = update_queue(
                self.queue[pid], energy_j, budget)

            # MAJ energie cumulee + estimation compute/epoch (EMA pour
            # lisser les mesures bruitees).
            self.energy_used[pid] = old_energy_used + energy_j
            if epochs_used > 0:
                new_e = compute_energy_j / float(epochs_used)
                old_e = self.energy_per_epoch.get(pid, 0.0)
                if old_e > 0:
                    alpha = 0.3
                    self.energy_per_epoch[pid] = (
                        (1.0 - alpha) * old_e + alpha * new_e
                    )
                else:
                    self.energy_per_epoch[pid] = new_e
                self.fixed_energy[pid] = max(0.0, energy_j - compute_energy_j)

            # Garde n_k pour le critere Lyapunov (ignore les drops n=0)
            n_k = float(mr.get(self.weighted_by_key, 0.0))
            if n_k > 0:
                self.num_examples[pid] = n_k

    def update_from_replies(self, replies, server_round):
        """Met a jour Q + energie pour chaque client a partir des replies."""
        metric_records = []
        for msg in replies:
            # Un client qui a plante renvoie un Message sans content : skip.
            if not getattr(msg, "has_content", lambda: True)():
                continue
            try:
                metric_records.append(next(iter(msg.content.metric_records.values())))
            except (ValueError, StopIteration, AttributeError):
                continue
        self.update_from_metric_records(metric_records, server_round)
