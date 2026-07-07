"""Modele (Net) + donnees CIFAR-10 (IID ou NON-IID Dirichlet) + partitionnement.

Le mode est controle par les params `partitioning` et `alpha` :
  * partitioning = "iid"             -> indices melanges puis parts egales
  * partitioning = "noniid"          -> Dirichlet par classe, tailles variables
  * partitioning = "noniid-balanced" -> Dirichlet labels, tailles equilibrees
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


DATA_ROOT = Path(os.environ.get("FL_DATA_ROOT", str(Path.home() / ".flwr_data")))
SEED = int(os.environ.get("FL_SEED", "42"))
VAL_RATIO = 0.2

_TRANSFORMS = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def set_seed(seed):
    """Force la reproducibilite. A appeler au debut de chaque process."""
    if seed is None or int(seed) < 0:
        return
    import random
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _group_norm(channels):
    """GroupNorm sans running stats, plus stable que BatchNorm en FL non-IID."""
    groups = 8
    while groups > 1 and int(channels) % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, int(channels))


class DepthwiseSeparableBlock(nn.Module):
    """Bloc MobileNet-like: depthwise conv + pointwise conv."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.use_residual = int(stride) == 1 and int(in_ch) == int(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            _group_norm(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _group_norm(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class Net(nn.Module):
    """CNN compact edge-IoT pour CIFAR-10 (~51k params)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            _group_norm(16),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableBlock(16, 32, stride=1),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 64, stride=1),
            DepthwiseSeparableBlock(64, 96, stride=2),
            DepthwiseSeparableBlock(96, 128, stride=1),
            DepthwiseSeparableBlock(128, 160, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Linear(160, 10),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


# Channel layouts par "model-name", utilises par CFL/submodel.py pour
# reduire dynamiquement la largeur du modele.
NET_CHANNELS = {
    "stem.out":       16,
    "block0.in":      16, "block0.out":     32,
    "block1.in":      32, "block1.out":     48,
    "block2.in":      48, "block2.out":     64,
    "block3.in":      64, "block3.out":     96,
    "block4.in":      96, "block4.out":     128,
    "block5.in":     128, "block5.out":     160,
    "classifier.in": 160, "classifier.out": 10,
}

BIGNET_CHANNELS = {
    "stem.out":       64,
    "block0.in":      64, "block0.out":     128,
    "block1.in":     128, "block1.out":     192,
    "block2.in":     192, "block2.out":     256,
    "block3.in":     256, "block3.out":     384,
    "block4.in":     384, "block4.out":     512,
    "block5.in":     512, "block5.out":     640,
    "classifier.in": 640, "classifier.out": 10,
}


class BigNet(nn.Module):
    """CNN ~4x plus large que Net (~734k params), meme architecture.

    Utile pour CFL : assez gros pour que les clients faibles aient besoin
    d'un sous-modele reduit.
    """

    def __init__(self):
        super().__init__()
        c = BIGNET_CHANNELS
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


class MobileNetV3Small(nn.Module):
    """MobileNetV3 Small (~1.5M params) adapte a CIFAR-10 (32x32).

    Adaptations : premier conv en stride=1 (au lieu de 2), classifier final
    a 10 classes, et BatchNorm remplace par GroupNorm (stabilite non-IID).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        if pretrained:
            base = mobilenet_v3_small(weights='IMAGENET1K_V1')
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            base = mobilenet_v3_small(weights=None, num_classes=num_classes)

        # Premier conv : stride 2 -> stride 1 (preserve la resolution 32x32).
        base.features[0][0] = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self._replace_bn_with_gn(base)

        self.model = base

    @staticmethod
    def _replace_bn_with_gn(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                groups = 8
                while groups > 1 and num_channels % groups != 0:
                    groups //= 2
                setattr(module, name, nn.GroupNorm(groups, num_channels))
            else:
                MobileNetV3Small._replace_bn_with_gn(child)

    def forward(self, x):
        return self.model(x)


# Dict factice pour satisfaire le registre : CFL ne supporte pas MobileNet
# (architecture trop differente de Net), submodel.py raise dans ce cas.
MOBILENET_CHANNELS = {
    "stem.out": 16,
}


# Registre central des modeles disponibles.
_MODEL_REGISTRY = {
    "net":       (Net,             NET_CHANNELS),
    "bignet":    (BigNet,          BIGNET_CHANNELS),
    "mobilenet": (MobileNetV3Small, MOBILENET_CHANNELS),
}


def get_model_class(name: str = "net"):
    """Retourne la CLASSE du modele pour un name donne (registre)."""
    key = str(name).lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"model-name inconnu : '{name}'. "
            f"Disponibles : {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[key][0]


def get_model_channels(name: str = "net"):
    """Retourne le dict de channels du modele (pour CFL submodel logic)."""
    key = str(name).lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"model-name inconnu : '{name}'. "
            f"Disponibles : {sorted(_MODEL_REGISTRY.keys())}"
        )
    return dict(_MODEL_REGISTRY[key][1])


def get_model(name: str = "net"):
    """Instancie le modele specifie. Helper le plus utilise par les apps."""
    cls = get_model_class(name)
    return cls()


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


_model_size_bytes_cache: dict = {}


def model_size_bytes(model_name: str = "net"):
    """Taille (octets) des parametres du modele. Memoize par model_name."""
    key = str(model_name).lower().strip()
    if key not in _MODEL_REGISTRY:
        key = "net"
    if key not in _model_size_bytes_cache:
        _model_size_bytes_cache[key] = sum(
            p.numel() * p.element_size() for p in get_model(key).parameters())
    return _model_size_bytes_cache[key]


_trainset = None
_testset = None


class HFCIFAR10(Dataset):
    """Fallback CIFAR-10 depuis Hugging Face quand torchvision ne telecharge pas."""

    def __init__(self, split, transform=None):
        from datasets import load_dataset

        self.ds = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform
        self.targets = [int(y) for y in self.ds["label"]]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img, label = item["img"], int(item["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


def _load_cifar10(train):
    try:
        return CIFAR10(root=str(DATA_ROOT), train=train, download=True,
                       transform=_TRANSFORMS)
    except Exception as exc:
        split = "train" if train else "test"
        print(f"[data] WARN CIFAR10 torchvision indisponible ({exc}); "
              f"fallback Hugging Face split={split}")
        return HFCIFAR10(split, transform=_TRANSFORMS)


def get_trainset():
    global _trainset
    if _trainset is None:
        _trainset = _load_cifar10(train=True)
    return _trainset


def get_testset():
    """Test set CIFAR-10 officiel (10k images equilibrees, jamais vu en train)."""
    global _testset
    if _testset is None:
        _testset = _load_cifar10(train=False)
    return _testset


# Proxy dataset pour FedAvg+KL : sous-ensemble stable du test CIFAR-10,
# pris hors des partitions clients pour eviter toute fuite train/eval.
_proxy_cache = {}


def get_proxy_subset(proxy_size: int = 2000, seed: int = 42):
    """Retourne (proxy_dataset, indices_list) stable pour FedAvg+KL.

    `proxy_size` = nombre d'images echantillonnees dans le test CIFAR-10.
    `seed` = graine du shuffle deterministe (les memes indices a chaque run).
    """
    key = (int(proxy_size), int(seed))
    if key in _proxy_cache:
        return _proxy_cache[key]
    test_ds = get_testset()
    total = len(test_ds)
    proxy_size = max(1, min(int(proxy_size), total))
    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(total)
    rng.shuffle(all_idx)
    indices = all_idx[:proxy_size].tolist()
    proxy_ds = Subset(test_ds, indices)
    _proxy_cache[key] = (proxy_ds, indices)
    return proxy_ds, indices


_eval_subset_cache = {}


def get_testset_excluding_indices(excluded_indices):
    """Test set CIFAR-10 sans les indices utilises comme proxy public.

    FedAvg+KL distille sur un proxy partage. Si ce proxy vient du test set, il ne
    doit pas aussi compter dans l'evaluation centrale, sinon l'accuracy serveur
    mesure partiellement des images vues via distillation.
    """
    excluded = frozenset(int(i) for i in (excluded_indices or []))
    if not excluded:
        return get_testset()
    key = tuple(sorted(excluded))
    if key not in _eval_subset_cache:
        test_ds = get_testset()
        keep = [i for i in range(len(test_ds)) if i not in excluded]
        _eval_subset_cache[key] = Subset(test_ds, keep)
    return _eval_subset_cache[key]


_parts_cache = {}


def _build_iid(num_partitions, seed):
    idx = np.arange(len(get_trainset()))
    np.random.default_rng(seed).shuffle(idx)
    return [p.tolist() for p in np.array_split(idx, num_partitions)]


def _build_dirichlet(num_partitions, alpha, seed):
    """Dirichlet par classe : pour chaque label, repartit ses indices entre clients
    selon proportions ~ Dirichlet(alpha)."""
    targets = np.asarray(get_trainset().targets)
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_partitions)]

    for label in np.unique(targets):
        label_idx = np.where(targets == label)[0]
        rng.shuffle(label_idx)
        proportions = rng.dirichlet([alpha] * num_partitions)
        counts = rng.multinomial(len(label_idx), proportions)
        start = 0
        for pid, c in enumerate(counts):
            parts[pid].extend(label_idx[start:start + c].tolist())
            start += c

    # Garantit que chaque client a au moins 1 exemple (donne par le plus gros)
    for pid in range(num_partitions):
        if not parts[pid]:
            donor = max(range(num_partitions), key=lambda i: len(parts[i]))
            if len(parts[donor]) > 1:
                parts[pid].append(parts[donor].pop())
    return parts


def _build_dirichlet_balanced(num_partitions, alpha, seed):
    """Dirichlet par client avec quotas egaux : garde le label skew mais
    donne a chaque client quasiment la meme quantite de donnees."""
    targets = np.asarray(get_trainset().targets)
    labels = np.unique(targets)
    rng = np.random.default_rng(seed)

    pools = {}
    for label in labels:
        label_idx = np.where(targets == label)[0]
        rng.shuffle(label_idx)
        pools[int(label)] = label_idx.tolist()

    quotas = [len(p) for p in np.array_split(np.arange(len(targets)),
                                             num_partitions)]
    proportions = rng.dirichlet([float(alpha)] * len(labels),
                                size=num_partitions)
    parts = [[] for _ in range(num_partitions)]

    while any(len(parts[pid]) < quotas[pid] for pid in range(num_partitions)):
        progressed = False
        for pid in rng.permutation(num_partitions):
            if len(parts[pid]) >= quotas[pid]:
                continue
            available_pos = [
                i for i, label in enumerate(labels) if pools[int(label)]
            ]
            if not available_pos:
                break

            probs = proportions[pid, available_pos].astype(float)
            if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
                probs = np.full(len(available_pos), 1.0 / len(available_pos))
            else:
                probs = probs / probs.sum()

            label_pos = int(rng.choice(available_pos, p=probs))
            label = int(labels[label_pos])
            parts[pid].append(pools[label].pop())
            progressed = True
        if not progressed:
            break

    return parts


def build_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    mode = str(partitioning).lower()
    if mode == "iid":
        return _build_iid(num_partitions, seed)
    if float(alpha) <= 0.0:
        raise ValueError("dirichlet-alpha doit etre > 0 pour partitionnement noniid")
    if mode == "noniid":
        return _build_dirichlet(num_partitions, float(alpha), seed)
    if mode == "noniid-balanced":
        return _build_dirichlet_balanced(num_partitions, float(alpha), seed)
    raise ValueError(
        f"partitioning={partitioning!r} invalide "
        "(attendu: 'iid'|'noniid'|'noniid-balanced')"
    )


def get_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    """Memoize build_partitions (les partitions sont stables pour un (n, mode, alpha, seed) donne)."""
    key = (int(num_partitions), str(partitioning).lower(), float(alpha), int(seed))
    if key not in _parts_cache:
        _parts_cache[key] = build_partitions(num_partitions, partitioning, alpha, seed)
    return _parts_cache[key]


def partition_sizes(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    return [len(p) for p in get_partitions(num_partitions, partitioning, alpha, seed)]


def clear_partitions_cache():
    """Vide le cache de partitions (utile pour les tests / re-runs in-process)."""
    _parts_cache.clear()


def _split_train_val(idx_list):
    """Coupe une liste d'indices en (train, val) selon VAL_RATIO."""
    n = len(idx_list)
    if n == 0:
        return [], []
    if n == 1:
        return idx_list, []
    val_size = min(max(1, int(n * VAL_RATIO)), n - 1)
    return idx_list[:-val_size], idx_list[-val_size:]


def _apply_data_hetero(tr_idx, pid, num_partitions, seed=SEED):
    """Tronque le train du client a une fraction tiree au hasard dans [0.2, 1.0].

    Le tirage est seede par pid (reproductible mais non correle au tier).
    """
    rng = np.random.default_rng(int(seed) + pid + 7919)
    keep = float(rng.uniform(0.2, 1.0))
    n_keep = max(1, int(len(tr_idx) * keep))
    return tr_idx[:n_keep]


def _make_loader_generator(pid, seed=-1):
    """Generator torch seede pour un shuffle DataLoader deterministe (None si seed < 0)."""
    if seed is None or int(seed) < 0:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed) + int(pid))
    return g


def load_data(pid, num_partitions, batch_size, data_hetero=0,
              partitioning="noniid", alpha=0.3, seed=-1, loader_seed=None):
    """Retourne (trainloader, valloader) pour le client `pid`.

    Si data_hetero=1, train est tronque selon `keep(pid)` pour simuler des
    tailles differentes. Le val reste complet pour comparabilite.
    `seed` controle uniquement le partitionnement; `loader_seed` controle le
    shuffle DataLoader. Ainsi seed=-1 garde bien un shuffle non-deterministe
    meme si `data-seed` fixe les partitions.
    """
    part_seed = SEED if seed is None or int(seed) < 0 else int(seed)
    dl_seed = loader_seed

    ds = get_trainset()
    idx = np.array(get_partitions(num_partitions, partitioning, alpha, part_seed)[pid])
    np.random.default_rng(part_seed + pid).shuffle(idx)
    tr, va = _split_train_val(idx.tolist())

    tr_full = list(tr)
    if int(data_hetero):
        tr = _apply_data_hetero(tr, pid, num_partitions, seed=part_seed)
    if not tr:
        tr = tr_full[:1] or va[:1]
    # va vide (cas n=1) : on retombe sur le train pour eviter un loader vide.
    if not va:
        va = tr[:1]

    gen = _make_loader_generator(pid, dl_seed)
    return (
        DataLoader(Subset(ds, tr), batch_size=batch_size,
                   shuffle=bool(tr), generator=gen),
        DataLoader(Subset(ds, va), batch_size=batch_size, shuffle=False),
    )
