"""Import helpers for Flower's isolated runtime app directory.

`flwr run .` copies the selected app into `~/.flwr/apps/...`. In this repo the
HFL-BEFL-CFL app reuses sibling packages (`cfl`, `hfl_befl`, `fl_common`).
Depending on the Flower packager, those siblings might not be installed in the
runtime app directory. This helper adds the source workspace package roots when
the run is launched from the repo root.
"""

from pathlib import Path
import sys


def ensure_workspace_packages():
    """Add local sibling package roots to sys.path if they exist."""
    candidates = []
    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(cwd.parents)
    here = Path(__file__).resolve()
    candidates.append(here.parent)
    candidates.extend(here.parents)

    for root in candidates:
        for rel in ("cfl", "hfl_befl", "fl_common"):
            path = root / rel
            if (path / rel / "__init__.py").exists():
                s = str(path)
                if s not in sys.path:
                    sys.path.insert(0, s)
