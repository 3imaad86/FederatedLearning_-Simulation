"""CFL : Customized Federated Learning (Wang et al. 2023).

Reference :
  "Towards Fairer and More Efficient Federated Learning via Multidimensional
   Personalized Edge Models" -- Wang, Guo, Zhang, Guo, Zhang, Zheng (2023)
  arXiv:2302.04464

Implementation pragmatique du repo : submodels par MASQUES de canaux
(prefix-keep), accuracy predictor MLP online (Algo 2), agregation avec
zero-pad expansion canal-par-canal (Algo 3).

Limitations par rapport au papier :
  * Pas de Once-For-All MobileNetV3 elastique (depth/kernel) -> width only.
  * Pas d'algorithme genetique pour la recherche -> random search S iters.
  * Pas de latency table mesuree -> proxy analytique latency ~ w^2 / tier.
"""

from pathlib import Path
import sys

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_COMMON = _WORKSPACE_ROOT / "fl_common"
if _LOCAL_COMMON.exists() and str(_LOCAL_COMMON) not in sys.path:
    sys.path.insert(0, str(_LOCAL_COMMON))
