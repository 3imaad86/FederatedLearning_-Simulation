"""Construction et alignement des submodels CFL (Wang et al. 2023).

Un submodel de `width_ratio` w garde les ceil(c*w) premiers canaux de
chaque couche du parent (prefix-keeping). Pour l'agregation, le serveur
zero-pad les canaux manquants et utilise un masque binaire pour faire
une moyenne ponderee canal par canal (width expansion).

Latency proxy : latency(w) ~ w^2 * base_latency (FLOPs Conv2d en w^2).
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl_common.data import (
    DepthwiseSeparableBlock,
    _group_norm,
    get_model_channels,
)


DEFAULT_CANDIDATE_WIDTHS = [0.25, 0.5, 0.75, 1.0]


def submodel_channels(width_ratio: float, model_name: str = "net") -> Dict[str, int]:
    """Channels de chaque couche du submodel pour un width donne.

    Convention `ceil(c * w)` (au moins 1 canal) ; la sortie classifier
    reste 10. Valide aussi que l'architecture parent est bien chainee.
    """
    parent = get_model_channels(model_name)
    if parent["stem.out"] != parent["block0.in"]:
        raise ValueError(
            f"submodel_channels({model_name}): stem.out={parent['stem.out']} "
            f"!= block0.in={parent['block0.in']}. Architecture non chainee."
        )
    for i in range(5):
        out_i = parent[f"block{i}.out"]
        in_next = parent[f"block{i+1}.in"]
        if out_i != in_next:
            raise ValueError(
                f"submodel_channels({model_name}): block{i}.out={out_i} != "
                f"block{i+1}.in={in_next}. Architecture non chainee."
            )
    if parent["block5.out"] != parent["classifier.in"]:
        raise ValueError(
            f"submodel_channels({model_name}): block5.out="
            f"{parent['block5.out']} != classifier.in="
            f"{parent['classifier.in']}. Architecture non chainee."
        )

    w = float(width_ratio)
    return {
        "stem.out":       max(1, math.ceil(parent["stem.out"] * w)),
        "block0.in":      max(1, math.ceil(parent["block0.in"] * w)),
        "block0.out":     max(1, math.ceil(parent["block0.out"] * w)),
        "block1.in":      max(1, math.ceil(parent["block1.in"] * w)),
        "block1.out":     max(1, math.ceil(parent["block1.out"] * w)),
        "block2.in":      max(1, math.ceil(parent["block2.in"] * w)),
        "block2.out":     max(1, math.ceil(parent["block2.out"] * w)),
        "block3.in":      max(1, math.ceil(parent["block3.in"] * w)),
        "block3.out":     max(1, math.ceil(parent["block3.out"] * w)),
        "block4.in":      max(1, math.ceil(parent["block4.in"] * w)),
        "block4.out":     max(1, math.ceil(parent["block4.out"] * w)),
        "block5.in":      max(1, math.ceil(parent["block5.in"] * w)),
        "block5.out":     max(1, math.ceil(parent["block5.out"] * w)),
        "classifier.in":  max(1, math.ceil(parent["classifier.in"] * w)),
        "classifier.out": int(parent["classifier.out"]),
    }


class SubNet(nn.Module):
    """Variant de Net avec channel counts parametrables.

    Meme architecture et memes noms de couches que le parent, seules les
    largeurs varient (mapping direct des state_dicts via prefix-keep).
    """

    def __init__(self, channels: Dict[str, int]):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c["stem.out"], 3, padding=1, bias=False),
            _group_norm(c["stem.out"]),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableBlock(c["block0.in"], c["block0.out"], stride=1),
            DepthwiseSeparableBlock(c["block1.in"], c["block1.out"], stride=2),
            DepthwiseSeparableBlock(c["block2.in"], c["block2.out"], stride=1),
            DepthwiseSeparableBlock(c["block3.in"], c["block3.out"], stride=2),
            DepthwiseSeparableBlock(c["block4.in"], c["block4.out"], stride=1),
            DepthwiseSeparableBlock(c["block5.in"], c["block5.out"], stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Linear(c["classifier.in"], c["classifier.out"]),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def build_submodel(width_ratio: float, model_name: str = "net") -> SubNet:
    """Construit un SubNet pour le width_ratio donne (parent = model_name)."""
    return SubNet(submodel_channels(width_ratio, model_name))


def _prefix_keep(t: torch.Tensor, dims_keep: Tuple[int, ...]) -> torch.Tensor:
    """Garde les `dims_keep[i]` premiers elements de la dimension i."""
    slicers = [slice(0, k) if k is not None else slice(None) for k in dims_keep]
    while len(slicers) < t.ndim:
        slicers.append(slice(None))
    return t[tuple(slicers)].contiguous()


def parent_to_submodel_state_dict(
    parent_sd: Dict[str, torch.Tensor],
    width_ratio: float,
    model_name: str = "net",
) -> Dict[str, torch.Tensor]:
    """Extrait les K premiers canaux du parent pour produire un submodel sd.

    Les noms de couches sont identiques entre parent et SubNet ; on tronque
    les bonnes dimensions de chaque tensor.
    """
    c = submodel_channels(width_ratio, model_name)
    sub_sd: Dict[str, torch.Tensor] = {}

    # ----- Stem -----
    # Conv 3 -> stem.out : (out=16, in=3, 3, 3) -> (stem.out, 3, 3, 3)
    sub_sd["features.0.weight"] = _prefix_keep(
        parent_sd["features.0.weight"], (c["stem.out"], None))
    # GroupNorm(stem.out)
    for suffix in ("weight", "bias"):
        sub_sd[f"features.1.{suffix}"] = _prefix_keep(
            parent_sd[f"features.1.{suffix}"], (c["stem.out"],))

    # ----- Blocks 0..5 (features.3..features.8) -----
    block_channels = [
        (3, c["block0.in"], c["block0.out"]),  # features.3 (block 0)
        (4, c["block1.in"], c["block1.out"]),  # features.4 (block 1)
        (5, c["block2.in"], c["block2.out"]),  # features.5 (block 2)
        (6, c["block3.in"], c["block3.out"]),  # features.6 (block 3)
        (7, c["block4.in"], c["block4.out"]),  # features.7 (block 4)
        (8, c["block5.in"], c["block5.out"]),  # features.8 (block 5)
    ]
    for f_idx, in_c, out_c in block_channels:
        # block.0 : Depthwise conv (in_ch, 1, kH, kW) groups=in_ch
        sub_sd[f"features.{f_idx}.block.0.weight"] = _prefix_keep(
            parent_sd[f"features.{f_idx}.block.0.weight"], (in_c, None))
        # block.1 : GN(in_ch)
        for suffix in ("weight", "bias"):
            sub_sd[f"features.{f_idx}.block.1.{suffix}"] = _prefix_keep(
                parent_sd[f"features.{f_idx}.block.1.{suffix}"], (in_c,))
        # block.3 : Pointwise conv (out_ch, in_ch, 1, 1)
        sub_sd[f"features.{f_idx}.block.3.weight"] = _prefix_keep(
            parent_sd[f"features.{f_idx}.block.3.weight"], (out_c, in_c))
        # block.4 : GN(out_ch)
        for suffix in ("weight", "bias"):
            sub_sd[f"features.{f_idx}.block.4.{suffix}"] = _prefix_keep(
                parent_sd[f"features.{f_idx}.block.4.{suffix}"], (out_c,))

    # ----- Classifier (Linear classifier.in -> 10) -----
    # Dropout n'a pas de params. La couche Linear est `classifier.1`.
    sub_sd["classifier.1.weight"] = _prefix_keep(
        parent_sd["classifier.1.weight"], (None, c["classifier.in"]))
    sub_sd["classifier.1.bias"] = _prefix_keep(
        parent_sd["classifier.1.bias"], (None,))

    return sub_sd


def _zero_pad_to(target_shape: Tuple[int, ...], source: torch.Tensor) -> torch.Tensor:
    """Place le tensor source en haut-gauche d'un tensor de zeros a target_shape."""
    out = torch.zeros(target_shape, dtype=source.dtype, device=source.device)
    slicers = tuple(slice(0, s) for s in source.shape)
    out[slicers] = source
    return out


def submodel_to_parent_zero_pad(
    sub_sd: Dict[str, torch.Tensor],
    parent_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Pour chaque cle commune, zero-pad le tensor submodel a la shape parent."""
    expanded: Dict[str, torch.Tensor] = {}
    for name, parent_t in parent_sd.items():
        if not parent_t.is_floating_point():
            expanded[name] = parent_t
            continue
        if name in sub_sd:
            expanded[name] = _zero_pad_to(parent_t.shape, sub_sd[name])
        else:
            # Cle absente cote client : zeros (exclue via le masque actif).
            expanded[name] = torch.zeros_like(parent_t)
    return expanded


def active_channel_mask(
    parent_sd: Dict[str, torch.Tensor],
    width_ratio: float,
    model_name: str = "net",
) -> Dict[str, torch.Tensor]:
    """Pour chaque tensor du parent : masque {0,1} des positions actives
    pour ce width_ratio, utilise par la moyenne ponderee canal par canal."""
    sub_geom = parent_to_submodel_state_dict(parent_sd, width_ratio, model_name)
    masks: Dict[str, torch.Tensor] = {}
    for name, parent_t in parent_sd.items():
        if not parent_t.is_floating_point():
            masks[name] = torch.ones_like(parent_t, dtype=torch.float32)
            continue
        if name in sub_geom:
            m = torch.zeros(parent_t.shape, dtype=torch.float32,
                            device=parent_t.device)
            slicers = tuple(slice(0, s) for s in sub_geom[name].shape)
            m[slicers] = 1.0
            masks[name] = m
        else:
            masks[name] = torch.zeros_like(parent_t, dtype=torch.float32)
    return masks


# Latence de base du full model (w=1.0) par tier hardware.
_BASE_LATENCY_PER_TIER = {0: 1.0, 1: 0.4, 2: 0.18}

# Budget latence max par tier : tier 0 tolere au plus w=0.5, tier 1 w=0.75,
# tier 2 w=1.0.
_LATENCY_BUDGETS = {0: 0.25, 1: 0.5, 2: 1.0}


def latency_proxy(width_ratio: float, hardware_tier: int) -> float:
    """Latence simulee d'un submodel de ratio w sur un tier (cout Conv2d en w^2)."""
    w = float(width_ratio)
    base = _BASE_LATENCY_PER_TIER.get(int(hardware_tier), 1.0)
    return base * w * w


def latency_budget(hardware_tier: int) -> float:
    """Budget latence pour un tier (cf. _LATENCY_BUDGETS)."""
    return _LATENCY_BUDGETS.get(int(hardware_tier), 1.0)


def feasible_widths(hardware_tier: int,
                    candidates: List[float] = None) -> List[float]:
    """Widths satisfaisant latency_proxy <= budget (fallback : le plus petit)."""
    cands = candidates if candidates is not None else DEFAULT_CANDIDATE_WIDTHS
    budget = latency_budget(hardware_tier)
    valid = [w for w in cands if latency_proxy(w, hardware_tier) <= budget]
    if not valid:
        return [min(cands)]
    return valid
