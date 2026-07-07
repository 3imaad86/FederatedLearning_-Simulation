"""KMeans-DRE : detecter si un sample proxy est proche des donnees locales du client.

Principe : on calcule des centroides KMeans sur la partition locale, puis un
sample x est declare In-Distribution (ID) si min_k ||x - c_k|| <= seuil.
Le seuil est le percentile (90e par defaut) des distances des donnees locales
a leurs centroides. Les features sont les pixels aplatis.
"""

import torch


def _flatten_images(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*H*W). Si deja 2D, retourne tel quel."""
    if x.dim() == 2:
        return x
    return x.reshape(x.size(0), -1)


def kmeans_torch(
    x: torch.Tensor,
    k: int,
    num_iters: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    """KMeans++ basique en PyTorch. Retourne les centroides (k, D).

    Si N <= k, chaque sample devient son propre centroide.
    """
    n = int(x.size(0))
    k = max(1, int(k))
    if n <= k:
        return x.clone()

    # Init KMeans++ : 1er centroide aleatoire, suivants ponderes par dist^2.
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    idx0 = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
    centroids = [x[idx0].clone()]
    for _ in range(k - 1):
        c_stack = torch.stack(centroids, dim=0)
        d2 = torch.cdist(x, c_stack).pow(2)
        min_d2 = d2.min(dim=1).values
        if float(min_d2.sum().item()) <= 0.0:
            probs = torch.ones_like(min_d2) / n
        else:
            probs = min_d2 / min_d2.sum()
        next_idx = int(
            torch.multinomial(probs, 1, generator=gen).item())
        centroids.append(x[next_idx].clone())
    centroids = torch.stack(centroids, dim=0)

    # Iterations de Lloyd
    for _ in range(int(num_iters)):
        d2 = torch.cdist(x, centroids).pow(2)
        assignments = d2.argmin(dim=1)
        new_centroids = centroids.clone()
        for j in range(k):
            members = x[assignments == j]
            if members.size(0) > 0:
                new_centroids[j] = members.mean(dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids


@torch.no_grad()
def fit_kmeans_per_class(
    loader,
    num_classes: int,
    clusters_per_class: int = 1,
    device: str = "cpu",
    num_iters: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    """Calcule `clusters_per_class` centroides KMeans pour chaque classe du loader.

    Une classe absente localement n'a pas de centroide : ses samples proxy
    seront vus comme OOD par ce client. Retourne un tensor (C_total, D).
    """
    by_class = {c: [] for c in range(int(num_classes))}
    feat_dim = None
    for x, y in loader:
        x_flat = _flatten_images(x).float()
        if feat_dim is None:
            feat_dim = int(x_flat.size(1))
        for c in range(int(num_classes)):
            mask = (y == c)
            if mask.any():
                by_class[c].append(x_flat[mask])

    centroids_list = []
    for c in range(int(num_classes)):
        chunks = by_class[c]
        if not chunks:
            continue
        class_x = torch.cat(chunks, dim=0).to(device)
        cents_c = kmeans_torch(
            class_x, k=int(clusters_per_class),
            num_iters=int(num_iters),
            seed=int(seed) + int(c) * 7919,
        )
        centroids_list.append(cents_c)

    if not centroids_list:
        return torch.zeros((0, feat_dim or 1), device=device)
    return torch.cat(centroids_list, dim=0)


@torch.no_grad()
def compute_id_threshold(
    centroids: torch.Tensor,
    loader,
    percentile: float = 90.0,
    device: str = "cpu",
) -> float:
    """Seuil T^ID = percentile des distances min des donnees locales aux centroides."""
    if int(centroids.size(0)) == 0:
        return float("inf")
    all_min_d = []
    for x, _ in loader:
        x_flat = _flatten_images(x).float().to(device)
        d = torch.cdist(x_flat, centroids)
        min_d = d.min(dim=1).values
        all_min_d.append(min_d)
    if not all_min_d:
        return float("inf")
    cat = torch.cat(all_min_d, dim=0)
    p = max(0.0, min(100.0, float(percentile))) / 100.0
    threshold = float(torch.quantile(cat, p).item())
    return threshold


@torch.no_grad()
def filter_proxy_id_mask(
    proxy_x: torch.Tensor,
    centroids: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Masque (B,) de 1.0 (ID) / 0.0 (OOD) pour un batch proxy."""
    if int(centroids.size(0)) == 0:
        return torch.zeros(int(proxy_x.size(0)), device=proxy_x.device)
    x_flat = _flatten_images(proxy_x).float().to(centroids.device)
    d = torch.cdist(x_flat, centroids)
    min_d = d.min(dim=1).values
    mask = (min_d <= float(threshold)).float()
    return mask
