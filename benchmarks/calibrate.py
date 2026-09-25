"""Back-compat entry for ``python benchmarks/calibrate.py ...``.

Prefer the installable ``semif-calibrate`` command from an editable checkout.
"""
from semif_phase1.calibration import *  # noqa: F401,F403
from semif_phase1.calibration import main

if __name__ == "__main__":
    main()
