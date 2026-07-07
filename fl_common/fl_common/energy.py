"""Modele d'energie simple pour edge IoT.

Consommation d'un client sur un round :
    E_compute = E_sample_epoch(tier) * num_examples * epochs
    E_comm    = P_comm(net_tier, link) * comm_time_s
    E_total   = E_compute + E_comm

Deux tables de puissance radio : WAN (liens cellulaires, FedAvg/...) et
LAN (WiFi/Ethernet local, HFL device-edge), le LAN consommant bien moins.
"""

# Energie de calcul par exemple et par epoch (J/sample/epoch).
# Le tier 0 (weak) consomme LE PLUS par sample : un CPU generaliste bas de
# gamme est moins efficace qu'un edge server avec NPU/GPU dedie.
DEFAULT_COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J = {
    0: 0.020,  # weak device   : SoC mobile generaliste
    1: 0.010,  # medium device : SoC milieu de gamme
    2: 0.005,  # strong device : edge server avec NPU/GPU
}
COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J = dict(DEFAULT_COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J)

# Puissance CPU/GPU legacy (Watts), fallback si num_examples/epochs absents.
DEFAULT_POWER_COMPUTE_W = {
    0: 1.5,
    1: 4.0,
    2: 7.0,
}
POWER_COMPUTE_W = dict(DEFAULT_POWER_COMPUTE_W)

# Puissance radio WAN (client <-> cloud) : 0 = LoRa, 1 = LTE, 2 = WiFi/5G.
DEFAULT_POWER_COMM_WAN_W = {
    0: 0.2,
    1: 2.0,
    2: 0.8,
}
POWER_COMM_WAN_W = dict(DEFAULT_POWER_COMM_WAN_W)

# Puissance radio LAN (device <-> edge local) : WiFi/Ethernet basse conso.
DEFAULT_POWER_COMM_LAN_W = {
    0: 0.4,
    1: 0.5,
    2: 0.3,
}
POWER_COMM_LAN_W = dict(DEFAULT_POWER_COMM_LAN_W)

# Puissance de la liaison edge <-> cloud (configurable, depend du materiel).
DEFAULT_EDGE_CLOUD_POWER_W = 5.0
EDGE_CLOUD_POWER_W = DEFAULT_EDGE_CLOUD_POWER_W

# Alias backward-compat : les callers sans `link_type` tombent sur la table WAN.
POWER_COMM_W = POWER_COMM_WAN_W

# Capacite batterie relative selon le tier compute.
# `befl-battery-j` represente la batterie d'un client medium.
BATTERY_TIER_MULTIPLIER = {
    0: 0.5,  # weak device: petite batterie
    1: 1.0,  # medium device: batterie de reference
    2: 2.0,  # strong device: batterie plus grande
}


def _cfg_float(cfg, key, default):
    """Lit un float dans un dict/ConfigRecord, avec fallback robuste."""
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def configure_energy_model(cfg):
    """Expose les constantes energie dans le run-config.

    Cles supportees :
      energy-compute-tier0-j / tier1 / tier2
      energy-comm-wan-tier0-w / tier1 / tier2
      energy-comm-lan-tier0-w / tier1 / tier2
      energy-edge-cloud-power-w
    """
    global EDGE_CLOUD_POWER_W
    for tier in (0, 1, 2):
        COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J[tier] = _cfg_float(
            cfg, f"energy-compute-tier{tier}-j",
            DEFAULT_COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J[tier])
        POWER_COMPUTE_W[tier] = _cfg_float(
            cfg, f"energy-compute-legacy-tier{tier}-w",
            DEFAULT_POWER_COMPUTE_W[tier])
        POWER_COMM_WAN_W[tier] = _cfg_float(
            cfg, f"energy-comm-wan-tier{tier}-w",
            DEFAULT_POWER_COMM_WAN_W[tier])
        POWER_COMM_LAN_W[tier] = _cfg_float(
            cfg, f"energy-comm-lan-tier{tier}-w",
            DEFAULT_POWER_COMM_LAN_W[tier])
    EDGE_CLOUD_POWER_W = _cfg_float(
        cfg, "energy-edge-cloud-power-w", DEFAULT_EDGE_CLOUD_POWER_W)


def _comm_power_table(link_type):
    """Selection de la table de puissance radio selon le type de lien."""
    return POWER_COMM_LAN_W if str(link_type).lower() == "lan" else POWER_COMM_WAN_W


REFERENCE_MODEL_FOR_ENERGY = "net"


def model_size_factor(model_name="net"):
    """Ratio de taille du modele par rapport a la reference (Net, ~51k params).

    Module l'energie compute proportionnellement aux FLOPs : Net -> 1.0,
    BigNet -> ~14.4. Modele inconnu -> 1.0 (no-op).
    """
    # Import local pour eviter un cycle d'import (data.py importe energy.py).
    from .data import model_size_bytes as _model_size_bytes
    try:
        ref = _model_size_bytes(REFERENCE_MODEL_FOR_ENERGY)
        actual = _model_size_bytes(model_name)
        if ref <= 0:
            return 1.0
        return float(actual) / float(ref)
    except Exception:
        return 1.0


def compute_compute_energy_j(tier, local_time_s, num_examples=None, epochs=None):
    """Energie de calcul deterministe.

    Si `num_examples` et `epochs` sont fournis, modele stable base sur les
    sample-epochs. Sinon, fallback legacy `P * local_time_s`.
    """
    if num_examples is not None and epochs is not None:
        e_unit = COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J.get(
            int(tier), COMPUTE_ENERGY_PER_SAMPLE_EPOCH_J[1])
        return float(e_unit) * float(num_examples) * float(epochs)

    p_comp = POWER_COMPUTE_W.get(int(tier), POWER_COMPUTE_W[1])
    return float(p_comp) * float(local_time_s)


def compute_energy_components(tier, net_tier, local_time_s, comm_time_s,
                              link_type="wan", num_examples=None, epochs=None,
                              compute_scale=1.0, model_name=None):
    """Retourne (compute_J, comm_J) separement.

    `compute_scale` module l'energie de calcul (utile pour CFL : un submodel
    a width w consomme ~w^2). `model_name` ajoute un facteur proportionnel
    a la taille du modele vs Net.
    """
    table = _comm_power_table(link_type)
    p_comm = table.get(int(net_tier), table[1])
    compute_j = compute_compute_energy_j(
        tier, local_time_s, num_examples=num_examples, epochs=epochs)
    compute_j = float(compute_j) * float(compute_scale)
    if model_name is not None:
        compute_j = compute_j * model_size_factor(model_name)
    comm_j = float(p_comm) * float(comm_time_s)
    return compute_j, comm_j


def compute_edge_cloud_energy_j(comm_time_s, n_links=1):
    """Energie simple de la liaison edge <-> cloud."""
    return float(EDGE_CLOUD_POWER_W) * float(comm_time_s) * float(n_links)


def compute_energy_j(tier, net_tier, local_time_s, comm_time_s, link_type="wan",
                     num_examples=None, epochs=None):
    """Energie totale (J) sur le round = compute + comm."""
    c_j, m_j = compute_energy_components(
        tier, net_tier, local_time_s, comm_time_s, link_type=link_type,
        num_examples=num_examples, epochs=epochs)
    return c_j + m_j


def battery_for_tier(base_battery_j, tier):
    """Capacite batterie effective selon le tier (<= 0 = unlimited)."""
    if base_battery_j <= 0:
        return 0.0
    return float(base_battery_j) * BATTERY_TIER_MULTIPLIER.get(int(tier), 1.0)
