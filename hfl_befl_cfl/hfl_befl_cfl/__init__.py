"""HFL-BEFL + CFL : combo IoT 3-couches.

Combine :
  - HFL              : hierarchique device-edge-cloud
  - BEFL             : Lyapunov queue + budget batterie (Liu 2022)
  - RL edge assign   : REINFORCE pour l'affectation client-edge
  - CFL              : submodels par width + accuracy predictor (Wang 2023)

Les 3 mecanismes adressent des dimensions ORTHOGONALES d'heterogeneite :
  - HFL  : reseau (LAN local vs WAN cloud)
  - BEFL : energie (batterie limitee tier 0/1, unlimited tier 2)
  - CFL  : compute (width du submodel adapte au tier)

LIMITATIONS :
  * FedStrag n'est PAS pris en charge avec CFL (un client late renvoie un
    submodel de N rounds avant, possiblement a un width different du round
    courant -> agregation ambigue). Le strategy emet un WARN si actif.
"""

from pathlib import Path
import sys

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for _rel in ("fl_common", "cfl", "hfl_befl"):
    _path = _WORKSPACE_ROOT / _rel
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
