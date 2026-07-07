# Federated Learning sur Edge IoT — Benchmark de 11 algorithmes

Banc d'essai complet de **Federated Learning (FL)** construit avec [Flower](https://flower.ai/) et PyTorch, pour comparer 11 algorithmes FL sur **CIFAR-10** dans des conditions réalistes d'**edge computing IoT** : données non-IID, hétérogénéité matérielle, réseaux contraints, budget énergétique et stragglers.

Chaque algorithme est un projet Flower indépendant qui partage un socle commun (`fl_common`) : même modèle, mêmes données, mêmes métriques — la comparaison est donc strictement équitable.

## Algorithmes implémentés

| Dossier | Algorithme | Idée principale | Référence |
|---|---|---|---|
| `fedavg/` | **FedAvg** | Moyenne pondérée des poids clients | McMahan et al. 2017 |
| `fedprox/` | **FedProx** | FedAvg + terme proximal `(mu/2)*‖w - w_global‖²` côté client | Li et al. 2018 |
| `fednova/` | **FedNova** | Normalise les updates par le nombre de pas SGD locaux (tau_eff) | Wang et al. 2020 |
| `scaffold/` | **SCAFFOLD** | Control variates (c_global / c_local) pour corriger le drift non-IID | Karimireddy et al. 2020 |
| `fedeve/` | **FedEve** | Filtre de Kalman entre momentum serveur et updates clients | Shen et al. 2025 |
| `cfl/` | **CFL** | Sous-modèles à largeur variable selon le hardware du client + accuracy predictor | Wang et al. 2023 |
| `fedavg_kl/` | **FedAvg+KL** | FedAvg + distillation KL via dataset proxy, filtrage KMeans-DRE (variante hybride inspirée d'EdgeFD, Liu et al. 2025) | — |
| `qfedavg/` | **q-FedAvg** | Pondération `F_k^q` : les clients à haute loss pèsent plus (fairness) | Li, Sanjabi, Smith 2019 |
| `fairfed/` | **FairFed** | Repondération vers le consensus de loss (adapté CIFAR-10) | Ezzeldin et al. 2023 |
| `hfl_befl/` | **HFL-BEFL** | FL hiérarchique (device → edge → cloud) + sélection BEFL par queue de Lyapunov + affectation d'edge par RL (REINFORCE) | Liu et al. 2022 (BEFL) |
| `hfl_befl_cfl/` | **HFL-BEFL-CFL** | Combinaison HFL + BEFL + RL + sous-modèles CFL | — |

## Fonctionnalités de simulation

Le socle commun `fl_common/` fournit des conditions expérimentales identiques pour tous les algorithmes :

- **Partitionnement des données** : IID, non-IID Dirichlet (`dirichlet-alpha`), ou non-IID à tailles équilibrées (`noniid-balanced`).
- **Hétérogénéité matérielle** : 3 tiers de clients (weak / medium / strong) avec nombre d'epochs différent (`epochs-heterogeneity`) et/ou tailles de données différentes (`data-heterogeneity`).
- **Simulation réseau** : profils LoRa / LTE / WiFi (WAN) et WiFi local / Ethernet (LAN pour le HFL), avec bande passante, RTT, jitter et pannes aléatoires.
- **Stragglers** : dropouts réseau, deadline de round (`round-deadline-s`), et **FedStrag** (les updates en retard sont bufferisées puis réinjectées avec un poids réduit selon leur staleness, au lieu d'être jetées).
- **Modèle d'énergie** : énergie de calcul par sample-epoch selon le tier + énergie radio selon le lien (WAN/LAN), ventilée compute/comm dans les CSV.
- **Batterie (BEFL)** : budget énergétique par client, queue de Lyapunov, sélection adaptative des participants et du nombre d'epochs.
- **Compression des updates** : quantification (`compression-quantization-bits` : 16/8/4) et sparsification (`compression-sparsity-ratio`), avec comptabilisation exacte des octets transmis.
- **Comptabilité des communications** : octets réellement transmis par round, séparés WAN (backbone cloud) / LAN (edge local) — SCAFFOLD compte naturellement ~2× FedAvg, CFL compte la taille des sous-modèles, etc.

## Modèles

Trois architectures CNN (GroupNorm au lieu de BatchNorm, plus stable en FL non-IID), sélectionnables via `model-name` :

| Nom | Paramètres | Usage |
|---|---|---|
| `net` (défaut) | ~51 k | Comparaison inter-algorithmes standard |
| `bignet` | ~734 k | Expériences CFL (les clients faibles ont besoin d'un sous-modèle) |
| `mobilenet` | ~1.5 M | Étude d'ablation architecture (MobileNetV3 Small adapté 32×32) |

## Structure du projet

```
.
├── fl_common/            # Socle commun (package Python partagé)
│   └── fl_common/
│       ├── data.py           # Modèles (Net/BigNet/MobileNet) + CIFAR-10 + partitionnement
│       ├── training.py       # Boucles train/test (+ variante SCAFFOLD)
│       ├── strategy.py       # Stratégies serveur (FedAvg, FedProx, FedNova, SCAFFOLD, FedEve, HFL)
│       ├── server_runner.py  # Boucle serveur commune (rounds, éval, logs, early stopping)
│       ├── client_helpers.py # Helpers clients (config, réseau, replies, éval locale)
│       ├── straggler.py      # Simulation réseau + dropouts
│       ├── energy.py         # Modèle d'énergie (compute + radio)
│       ├── compression.py    # Quantification + sparsification des updates
│       └── metrics.py        # Métriques (accuracy, fairness JFI, convergence) + CSV
├── fedavg/               # Un projet Flower par algorithme
├── fedprox/
├── fednova/
├── scaffold/
├── fedeve/
├── cfl/
├── fedavg_kl/
├── qfedavg/
├── fairfed/
├── hfl_befl/
├── hfl_befl_cfl/
├── plot_results.py       # Génère les graphiques de comparaison (PNG dans plots/)
└── plots/                # Graphiques générés
```

Chaque dossier d'algorithme contient un `pyproject.toml` (config Flower + hyperparamètres), un `client_app.py` et un `server_app.py`.

## Installation

Prérequis : Python ≥ 3.10.

```bash
git clone <url-du-repo>
cd <repo>

# Environnement virtuel (ou conda)
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# Installe le socle commun puis l'algorithme voulu
pip install -e fl_common
pip install -e fedavg            # ou fedprox, scaffold, hfl_befl, ...
```

Dépendances principales (installées via `fl_common`) : `flwr[simulation] >= 1.26`, `torch 2.8`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `datasets`.

CIFAR-10 est téléchargé automatiquement au premier lancement (torchvision, avec fallback Hugging Face). Le cache est placé dans `~/.flwr_data` (modifiable via la variable d'environnement `FL_DATA_ROOT`).

## Lancer une expérience

Depuis le dossier de l'algorithme :

```bash
cd fedavg
flwr run . --stream \
  --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 num-clients=10 partitioning='noniid' dirichlet-alpha=0.3 seed=42"
```

> `num-supernodes` (nombre de clients simulés) doit être égal à `num-clients`.

Avec GPU, on peut paralléliser les clients :

```bash
flwr run . --stream \
  --federation-config "num-supernodes=10 client-resources-num-cpus=2 client-resources-num-gpus=0.1" \
  --run-config "num-server-rounds=30 num-clients=10"
```

### Exemples par algorithme

```bash
# FedProx : régler le terme proximal
cd fedprox && flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 mu=0.01"

# SCAFFOLD : exige momentum=0 (vanilla SGD)
cd scaffold && flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 scaffold-server-lr=1.0"

# FedNova : intéressant avec hétérogénéité d'epochs (1/3/6 selon le tier)
cd fednova && flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 epochs-heterogeneity=1"

# FedEve : participation partielle (régime cross-device)
cd fedeve && flwr run . --stream --federation-config "num-supernodes=100" \
  --run-config "num-server-rounds=50 num-clients=100 fraction-train=0.1 local-epochs=1"

# CFL : nécessite l'hétérogénéité hardware et gagne à utiliser BigNet
cd cfl && flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 hardware-heterogeneity=1 model-name='bignet'"

# FedAvg+KL : distillation via proxy
cd fedavg_kl && flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 fedavgkl-distill-lambda=0.5 fedavgkl-proxy-size=2000"

# HFL-BEFL : hiérarchique + batterie + stragglers + RL
cd hfl_befl && flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=60 hfl-num-edges=3 hfl-local-steps=3 befl-battery-j=5000 epochs-heterogeneity=1 straggler-sim=1 round-deadline-s=60 hfl-rl-edge-assignment=1"
```

## Options de configuration communes

Toutes les options se passent via `--run-config` (défauts dans le `pyproject.toml` de chaque algo) :

| Clé | Défaut | Description |
|---|---|---|
| `num-server-rounds` | 10 | Nombre de rounds FL |
| `num-clients` | 10 | Nombre de clients (= `num-supernodes`) |
| `local-epochs` | 2 | Epochs locaux par round |
| `learning-rate` / `batch-size` / `momentum` | 0.01 / 32 / 0.0 | Hyperparamètres SGD |
| `partitioning` | `noniid` | `iid`, `noniid` (Dirichlet) ou `noniid-balanced` |
| `dirichlet-alpha` | 0.3 | Concentration Dirichlet (petit = très non-IID) |
| `data-heterogeneity` | 0 | 1 = tailles de partitions aléatoires (20–100 %) |
| `epochs-heterogeneity` | 0 | 1 = epochs par tier (weak 1 / medium 3 / strong 6) |
| `hardware-heterogeneity` | = epochs-het. | 1 = tiers hardware sans changer les epochs (pour CFL) |
| `fraction-train` | 1.0 | Fraction de clients échantillonnés par round |
| `model-name` | `net` | `net`, `bignet` ou `mobilenet` |
| `seed` / `data-seed` | -1 / 42 | Seed runtime (reproductibilité) / seed du partitionnement |
| `straggler-sim` | 0 | 1 = pannes réseau + délais simulés (HFL-BEFL) |
| `round-deadline-s` | 0 | Deadline de round (0 = désactivée) |
| `comm-size-ratio` | 1.0 | Ratio de compression manuel des communications |
| `compression-quantization-bits` | 32 | 16 / 8 / 4 = quantification des updates (uplink) |
| `compression-sparsity-ratio` | 0.0 | Fraction des poids mis à zéro (uplink) |
| `sim-model-mb` | 0 | Taille de modèle simulée en MB (0 = taille réelle) |
| `early-stopping-patience` | 0 | Rounds sans amélioration avant arrêt (0 = off) |
| `energy-*` | — | Constantes du modèle d'énergie (voir `fl_common/energy.py`) |

Chaque algorithme ajoute ses propres clés (`mu`, `qfedavg-q`, `fedeve-server-lr`, `cfl-*`, `fedavgkl-*`, `hfl-*`, `befl-*`, `fedstrag-*`…) — voir le `pyproject.toml` du dossier concerné.

## Résultats et métriques

Chaque run écrit des CSV dans `<algo>/results/` :

| Fichier | Contenu |
|---|---|
| `metrics_global.csv` | Une ligne par round : accuracy/loss serveur (test set CIFAR-10), macro recall/F1, fairness par classe (JFI, worst-class), coût comm WAN/LAN (MB), temps de round, énergie (totale, compute/comm, par tier), diagnostics par algo (tau_eff, normes SCAFFOLD, gain de Kalman, widths CFL, batterie/queue BEFL…) |
| `metrics_per_class.csv` | Accuracy par classe et par round |
| `metrics_clients.csv` | Détail par client et par round (n, epochs, tier, réseau, temps, énergie, batterie, drops) |
| `metrics_summary.csv` | Synthèse : rounds-to-target (50/70/90 %), fairness de participation, et **ressources-to-target** (énergie/comm/temps cumulés pour atteindre 30/50/70 % d'accuracy) |
| `metrics_participation.csv` | Nombre de participations par client |
| `final_model.pt` | Poids du modèle global final |

L'évaluation est faite **côté serveur** sur le test set CIFAR-10 officiel (10 k images) : en non-IID, les métriques locales des clients surestiment l'accuracy (chaque client overfitte sa partition) — le gap serveur/local est d'ailleurs loggé comme mesure du biais non-IID.

## Graphiques de comparaison

Après un ou plusieurs runs :

```bash
python plot_results.py                    # tous les algos ayant des résultats
python plot_results.py fedavg scaffold    # sélection
```

Produit dans `plots/` : courbes d'accuracy/loss, coût de communication cumulé, énergie (totale et ventilée compute/radio), temps de round, heatmaps d'accuracy par classe, barres rounds-to-target, fairness inter-clients, consommation par client, etc.

## Notes de conception

- **Comparaison équitable** : mêmes partitions de données (seed dédié `data-seed`), même modèle, même éval serveur et mêmes métriques pour les 11 algorithmes.
- **Coûts mesurés, pas estimés** : les octets comptés sont ceux des messages Flower réellement transmis. SCAFFOLD paie ses control variates, FedAvg+KL paie ses logits, CFL ne paie que ses sous-modèles.
- **FedAvg+KL n'est pas EdgeFD** : c'est une variante hybride qui garde l'agrégation FedAvg des poids et ajoute la distillation KL comme régularisation ; le papier EdgeFD original (Liu et al. 2025) n'agrège pas les poids et suppose des modèles hétérogènes.
- **Contraintes théoriques respectées** : SCAFFOLD et FedNova refusent `momentum != 0` ; des avertissements sont émis quand une configuration sort du cadre théorique d'un algorithme (ex. FedAvg avec epochs hétérogènes, q-FedAvg multi-epoch).
- **Reproductibilité** : `seed >= 0` fixe l'init des modèles, le shuffle des DataLoaders, la simulation réseau, la policy RL et la recherche CFL.

## Licence

Apache-2.0
