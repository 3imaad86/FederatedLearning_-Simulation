"""FedEve : filtre de Kalman serveur pour compenser period drift et
client drift en cross-device FL (Shen et al. 2025, arXiv:2508.14539)."""
"""FedEve Flower app."""

from pathlib import Path
import sys

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_COMMON = _WORKSPACE_ROOT / "fl_common"
if _LOCAL_COMMON.exists() and str(_LOCAL_COMMON) not in sys.path:
    sys.path.insert(0, str(_LOCAL_COMMON))
