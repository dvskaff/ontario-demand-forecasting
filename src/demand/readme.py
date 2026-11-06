"""Rewrites the results block in README.md after each run."""

from __future__ import annotations

import re
from pathlib import Path


def results_block(s: dict) -> str:
    b, live = s["backtest"], s["live"]
    lines = [
        f"_IESO demand through {s['data_end']} · champion model v{s['champion']['version']} "
        f"(data through {s['champion']['data_end']}) · last run {s['built_at']}_",
        "",
        f"**Next-day forecast ({s['target_day']}):** peak {s['peak']['p50']:,.0f} MW at hour ending "
        f"{s['peak_hour_ending']}:00 EST (80% interval {s['peak']['p10']:,.0f}–{s['peak']['p90']:,.0f} MW)",
        "",
        "| Accuracy | MAPE | Daily peak error | 80% interval coverage* |",
        "|---|---|---|---|",
        f"| Backtest, LightGBM quantile ({b['months']} monthly folds) | {b['model']['mape']:.2f}% | "
        f"{b['model']['peak_ape']:.2f}% | {b['model']['coverage_80']:.0f}% |",
        f"| Baseline: same hour last week | {b['naive_week']['mape']:.2f}% | {b['naive_week']['peak_ape']:.2f}% | – |",
        f"| Baseline: same hour yesterday | {b['naive_day']['mape']:.2f}% | {b['naive_day']['peak_ape']:.2f}% | – |",
    ]
    if live.get("days"):
        lines.append(f"| **Live**, issued forecasts, last {live['days']} days | {live['mape']:.2f}% | "
                     f"{live['peak_ape']:.2f}% | {live['coverage_80']:.0f}% |")  # fmt: skip
    lines += [
        "",
        "_*Backtest coverage is before the conformal calibration of the interval; the live row is after._",
    ]
    rt = s.get("retrain")
    lines += ["", f"**Registry:** {s['versions']} model version(s) in MLflow"
              + (f"; this run retrained ({rt['trigger']}): v{rt['version']} {'promoted' if rt['promoted'] else 'not promoted'}"
                 f" ({rt['decision']})" if rt else "")]  # fmt: skip
    d = s["drift"]
    if d.get("available"):
        lines[-1] += (
            f" · **Drift (Evidently, vs. same weeks last year):** {d['drifted']} of {d['n_columns']} monitored columns drifted"
        )
    lines += ["", f"**API image:** `docker run -p 8000:8000 {s['image']}:latest`"]
    return "\n".join(lines)


def update_readme(readme: Path, summary: dict) -> None:
    text = readme.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)"
    readme.write_text(
        re.sub(pattern, lambda m: m.group(1) + results_block(summary) + m.group(2), text, flags=re.S)
    )
