#!/usr/bin/env python
"""Convenience entry point: run SAIP exactly as ``python -m saip`` would.

    python scripts/run_saip.py --config configs/activitynet.yaml \
        --manifest manifest.json --out-dir runs/activitynet

The ``-m`` form is the canonical one; this file exists so that the repository can
also be driven from environments that only invoke scripts directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saip.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
