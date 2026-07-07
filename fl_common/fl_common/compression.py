"""Compression des poids envoyes par les clients (quantization + sparsification).

Options dans run-config :
  - compression-quantization-bits : 32 (off) | 16 | 8 | 4
  - compression-sparsity-ratio    : 0.0 (off) | 0.5 | 0.9 ...

Non applique a SCAFFOLD (les control variates ne doivent pas etre compresses).
"""

import torch


def quantize_tensor(t, bits):
    """Quantifie un tensor float sur `bits` bits (les valeurs restent en fp32)."""
    if int(bits) >= 32 or not t.is_floating_point() or t.numel() == 0:
        return t
    max_abs = float(t.abs().max().item())
    if max_abs == 0.0:
        return t
    levels = (1 << (int(bits) - 1)) - 1   # ex: bits=8 -> [-127, 127]
    scale = max_abs / levels
    return (torch.round(t / scale).clamp(-levels, levels) * scale).to(t.dtype)


def sparsify_tensor(t, ratio):
    """Met a zero les `ratio` (ex: 0.9 = 90%) poids les plus petits en valeur absolue."""
    r = float(ratio)
    if r <= 0.0 or not t.is_floating_point() or t.numel() == 0:
        return t
    if r >= 1.0:
        return torch.zeros_like(t)
    n = t.numel()
    k = max(1, int(round(n * (1.0 - r))))  # nb d'elements a garder
    if k >= n:
        return t
    abs_t = t.abs().flatten()
    # kthvalue retourne le k-eme plus petit -> on veut le k-eme plus grand
    threshold = abs_t.kthvalue(n - k + 1).values
    mask = (t.abs() >= threshold).to(t.dtype)
    return t * mask


def apply_compression(state_dict, bits=32, sparsity=0.0,
                      skip_prefixes=None):
    """Applique sparsification puis quantification a chaque tensor float.

    `skip_prefixes` : prefixes de cles a ne pas toucher (ex: "__dc__" pour SCAFFOLD).
    """
    bits = int(bits)
    sparsity = float(sparsity)
    if bits >= 32 and sparsity <= 0.0:
        return state_dict
    if skip_prefixes is None:
        skip_prefixes = ()
    out = {}
    for name, t in state_dict.items():
        if any(name.startswith(p) for p in skip_prefixes):
            out[name] = t
            continue
        if not t.is_floating_point():
            out[name] = t
            continue
        t2 = t
        if sparsity > 0.0:
            t2 = sparsify_tensor(t2, sparsity)
        if bits < 32:
            t2 = quantize_tensor(t2, bits)
        out[name] = t2
    return out


def effective_size_ratio(bits=32, sparsity=0.0):
    """Ratio bytes compresses / bytes originaux = (bits/32) * (1 - sparsity)."""
    bit_ratio = float(bits) / 32.0 if int(bits) < 32 else 1.0
    sparse_ratio = max(0.0, 1.0 - float(sparsity))
    return bit_ratio * sparse_ratio
