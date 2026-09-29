"""Run the complete experiment suite and write plots, GIFs and metrics to results/.

Usage::

    python scripts/run_all.py            # everything (a few minutes on 4 CPU cores)
    python scripts/run_all.py --quick    # smoke test: one scenario, no robustness, no GIFs
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from quadrotor.control.base import hover_reference
from quadrotor.experiments import CASES, CONTROLLERS, make_controller, run_case, step_response
from quadrotor.params import QuadParams
from quadrotor.planning.scenarios import SCENARIOS
from quadrotor.plotting import (
    animate,
    plot_errors,
    plot_estimation,
    plot_metric_bars,
    plot_step,
    plot_trajectories,
)
from quadrotor.sim import Simulation

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def _job(args: tuple) -> tuple:
    scenario, controller, case, estimator, keep = args
    summary, res = run_case(scenario, controller, case, estimator=estimator, keep_result=keep)
    return args, summary, res


def _step_job(controller: str) -> dict:
    return step_response(controller)


def markdown_table(rows: list[dict], columns: list[str], headers: list[str]) -> str:
    """Render dict rows as a GitHub markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return "\n".join(lines)


def build_markdown(rows: list[dict], steps: list[dict], scenarios: list[str]) -> str:
    """Result tables used in results/metrics.md and README.md."""
    out = []
    out.append("#### Śledzenie trajektorii (estymata ESKF, bez zakłóceń)\n")
    nominal = [r for r in rows if r["case"] == "nominal" and r["estimator"] == "ekf"]
    cols = ["scenario", "controller", "rmse_pos_m", "max_err_m", "energy_J", "gate_miss_m"]
    hdr = [
        "Scenariusz",
        "Regulator",
        "RMSE [m]",
        "Błąd max [m]",
        "Energia [J]",
        "Chybienie bramki [m]",
    ]
    out.append(markdown_table(nominal, cols, hdr))
    ideal = [r for r in rows if r["estimator"] == "truth"]
    if ideal:
        out.append("\n#### Śledzenie trajektorii z idealną informacją o stanie (bez EKF)\n")
        out.append(markdown_table(ideal, cols, hdr))
    out.append("\n#### Odporność: RMSE śledzenia [m] (estymata ESKF)\n")
    robust = []
    for s, c in product(scenarios, CONTROLLERS):
        row = {"scenario": s, "controller": c}
        for case in CASES:
            m = [
                r
                for r in rows
                if (r["scenario"], r["controller"], r["case"], r["estimator"])
                == (s, c, case, "ekf")
            ]
            if m:
                row[case] = "**FAIL**" if m[0]["failed"] else m[0]["rmse_pos_m"]
        robust.append(row)
    present = [c for c in CASES if any(c in r for r in robust)]
    out.append(
        markdown_table(
            robust, ["scenario", "controller", *present], ["Scenariusz", "Regulator", *present]
        )
    )
    out.append("\n#### Odpowiedź na skok 1 m (po przekątnej, stan idealny)\n")
    out.append(
        markdown_table(
            steps,
            ["controller", "settling_time_s", "overshoot_m", "energy_J"],
            ["Regulator", "Czas ustalania (5 cm) [s]", "Przeregulowanie [m]", "Energia [J]"],
        )
    )
    return "\n".join(out) + "\n"


def update_readme(md: str) -> None:
    """Insert the tables between the RESULTS markers of README.md (if present)."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text()
    pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.S)
    if pat.search(text):
        readme.write_text(pat.sub(lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", text))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke test (one scenario, no GIFs)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    scenarios = ["figure8"] if args.quick else list(SCENARIOS)
    cases = ["nominal"] if args.quick else CASES
    jobs = [
        (s, c, case, "ekf", case == "nominal")
        for s, c, case in product(scenarios, CONTROLLERS, cases)
    ]
    if not args.quick:
        jobs += [(s, c, "nominal", "truth", False) for s, c in product(scenarios, CONTROLLERS)]

    print(f"running {len(jobs)} simulations on {args.workers} workers ...", flush=True)
    nominal_results: dict[str, dict] = {s: {} for s in scenarios}
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (a, summary, res) in enumerate(ex.map(_job, jobs), 1):
            rows.append(summary.as_row())
            if res is not None:
                nominal_results[a[0]][a[1]] = res
            print(
                f"  [{i}/{len(jobs)}] {a[0]:16s} {a[1]:10s} {a[2]:16s} {a[3]:5s} "
                f"rmse={summary.rmse:.3f} m",
                flush=True,
            )
        steps = list(ex.map(_step_job, CONTROLLERS))

    with open(RESULTS / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS / "step_response.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(steps[0]))
        w.writeheader()
        w.writerows(steps)
    md = build_markdown(rows, steps, scenarios)
    (RESULTS / "metrics.md").write_text(md)
    if not args.quick:
        update_readme(md)

    print("plotting ...", flush=True)
    params = QuadParams.from_yaml()
    for s in scenarios:
        sc = SCENARIOS[s]()
        plot_trajectories(sc, nominal_results[s], RESULTS / f"trajectory_{s}.png")
        plot_errors(sc, nominal_results[s], RESULTS / f"error_{s}.png")
    plot_estimation(
        nominal_results["figure8"]["Geometric"],
        RESULTS / "estimation_figure8.png",
        "ESKF, figure-8, geometric controller",
    )
    ekf_rows = [r for r in rows if r["estimator"] == "ekf"]
    plot_metric_bars(ekf_rows, RESULTS / "robustness.png", cases, CONTROLLERS)
    step_data = {}
    for c in CONTROLLERS:  # cheap: re-simulate steps with ground truth to plot the curves
        tgt = np.array([1.0, 1.0, 1.0])
        r = Simulation(
            params,
            make_controller(c, params),
            lambda t, tgt=tgt: hover_reference(tgt if t >= 0.5 else np.zeros(3)),
        ).run(5.5)
        step_data[c] = (r.t - 0.5, np.linalg.norm(r.x[:, :3] - tgt, axis=1))
    plot_step(step_data, RESULTS / "step_response.png")

    if not args.quick and not args.no_gif:
        for s in ("figure8", "slalom", "helix"):
            print(f"animating {s} ...", flush=True)
            animate(SCENARIOS[s](), nominal_results[s], params, RESULTS / f"{s}.gif")
    print(f"done in {time.time() - t0:.0f} s; results in {RESULTS}")


if __name__ == "__main__":
    main()
