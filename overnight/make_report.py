"""Rebuild the readable report for a run at any time (e.g. if the terminal was closed).

    .venv\\Scripts\\python overnight\\make_report.py                 # latest run
    .venv\\Scripts\\python overnight\\make_report.py <run folder>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor_lib import latest_run_dir, make_report  # noqa: E402

run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run_dir()
if run is None:
    sys.exit("no runs found under overnight/runs")
print(make_report(run))
