"""Simulation stragglers : reseau variable + dropouts aleatoires (edge IoT)."""

import os
import random as _rnd

SEED = int(os.environ.get("FL_SEED", "42"))

# tier : (bw_mbps, rtt_s, jitter_s, p_drop_par_round)
NET_TIERS = {
    0: (0.5,  0.8,  0.3,  0.15),   # faible (LoRa / 2G)        -> 15% dropout
    1: (5.0,  0.2,  0.05, 0.05),   # moyen  (LTE smartphone)   ->  5% dropout
    2: (50.0, 0.03, 0.01, 0.01),   # fort   (WiFi edge gateway)->  1% dropout
}
NET_TIER_WEIGHTS = [0.4, 0.4, 0.2]


def _resolve_seed(seed):
    """Seed du run si fourni (>= 0), sinon la valeur FL_SEED."""
    if seed is None:
        return SEED
    try:
        value = int(seed)
    except (TypeError, ValueError):
        return SEED
    return SEED if value < 0 else value


def mbps_transfer_time_s(model_mb, bandwidth_mbps):
    """Temps pour transferer `model_mb` MB sur `bandwidth_mbps` Mbps."""
    bw = float(bandwidth_mbps)
    if bw <= 0.0:
        raise ValueError("bandwidth_mbps doit etre > 0")
    return (float(model_mb) * 8.0) / bw


def roundtrip_delay_s(model_mb, bandwidth_mbps, rtt_s):
    """Download + upload du modele, plus RTT."""
    return 2.0 * mbps_transfer_time_s(model_mb, bandwidth_mbps) + float(rtt_s)


def network_profile(pid, seed=None):
    """Tier reseau stable du client (meme pid -> meme tier a tous les rounds)."""
    rng = _rnd.Random(_resolve_seed(seed) + int(pid))
    tier = rng.choices([0, 1, 2], weights=NET_TIER_WEIGHTS)[0]
    bw, rtt, jitter, pdrop = NET_TIERS[tier]
    return tier, bw, rtt, jitter, pdrop


def simulate_comm_delay(pid, model_mb, round_idx, seed=None):
    """Simule la communication d'un round. Retourne (tier, delay_s) ou (tier, None) si dropout."""
    seed_value = _resolve_seed(seed)
    tier, bw, rtt, jitter, pdrop = network_profile(pid, seed_value)
    rng = _rnd.Random(seed_value + int(pid) * 1000 + int(round_idx))
    if rng.random() < pdrop:
        return tier, None
    delay = roundtrip_delay_s(model_mb, bw, rtt) + abs(rng.gauss(0.0, jitter))
    return tier, delay
