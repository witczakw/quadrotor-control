"""Re-create only the 3D GIF animations (all controllers, nominal case, ESKF in the loop)."""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

from quadrotor.experiments import CONTROLLERS, run_case
from quadrotor.params import QuadParams
from quadrotor.planning.scenarios import SCENARIOS
from quadrotor.plotting import animate

RESULTS = Path(__file__).resolve().parents[1] / "results"
warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def _run(args: tuple[str, str]):
    scenario, controller = args
    return args, run_case(scenario, controller, "nominal", keep_result=True)[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenarios", nargs="*", default=["figure8", "slalom", "helix"])
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    params = QuadParams.from_yaml()
    out: dict[str, dict] = {s: {} for s in args.scenarios}
    with ProcessPoolExecutor(4) as ex:
        for (s, c), res in ex.map(_run, list(product(args.scenarios, CONTROLLERS))):
            out[s][c] = res
    for s in args.scenarios:
        print(f"animating {s} ...", flush=True)
        animate(SCENARIOS[s](), out[s], params, RESULTS / f"{s}.gif", fps=args.fps)


if __name__ == "__main__":
    main()
