"""Weather what-if: a page (site/whatif.html) where anyone can make tomorrow warmer or colder and watch the champion
model's forecast change, with the model itself running in the browser (ONNX Runtime Web).

The pipeline exports the next day's 24 feature rows, the 23 hours of temperature before the day, and the champion's
three ONNX models. Moving the slider shifts the day's temperatures; the page rebuilds the temperature features with
the same formulas as features.build_features (mirrored by `shift`, which the tests check against the real feature
code), runs the three quantile models and turns their outputs into MW exactly as features.to_demand does.
Demand history and the calendar stay the same, so only the weather changes.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, LEVEL, inference_features

log = logging.getLogger(__name__)
RUNTIME = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
TEMP_COLUMNS = ["temp", "heating_degrees", "cooling_degrees", "temp_roll24", "temp_day_mean", "temp_day_max"]
MAX_SHIFT = 10  # °C either way


def shift(features: pd.DataFrame, prev23: list[float], temps: list[float], delta: float) -> pd.DataFrame:
    """The day's feature rows with every temperature `delta` degrees warmer (the page does the same in JavaScript).
    prev23: the 23 hourly temperatures before the day (they feed the 24-hour rolling mean, and are not shifted)."""
    t = np.asarray(temps, dtype=float) + delta
    window = np.concatenate([np.asarray(prev23, dtype=float), t])
    out = features.copy()
    out["temp"] = t
    out["heating_degrees"] = np.clip(15.5 - t, 0, None)
    out["cooling_degrees"] = np.clip(t - 18, 0, None)
    out["temp_roll24"] = [window[h : h + 24].mean() for h in range(24)]
    out["temp_day_mean"] = t.mean()
    out["temp_day_max"] = t.max()
    return out


def export(cfg, frame: pd.DataFrame, target: pd.Timestamp, champ: dict, fc: pd.DataFrame) -> dict:
    """Copy the champion's ONNX models into the site and write data/whatif.json. Returns the exported data."""
    site = cfg.site_dir
    hours = pd.date_range(target, periods=24, freq="h")
    temps = frame["temp_c"].reindex(hours).astype(float).tolist()
    history = frame[frame.index < target]
    feats = inference_features(history, target.date(), temps)
    prev23 = history["temp_c"].reindex(pd.date_range(target - pd.Timedelta(hours=23), periods=23, freq="h"))
    onnx_dir = Path(champ["onnx_dir"])
    (site / "model").mkdir(parents=True, exist_ok=True)
    for f in ("p10.onnx", "p50.onnx", "p90.onnx"):
        shutil.copy(onnx_dir / f, site / "model" / f)
    spec_file = onnx_dir / "model.json"
    spec = json.loads(spec_file.read_text()) if spec_file.exists() else {"target": "absolute"}
    # history of what the model has seen: the warmest and coldest hours in the training data (trees can't
    # extrapolate beyond them, so the page says so when a scenario goes past them)
    seen = frame["temp_c"][frame.index < target].dropna()
    data = {
        "target_day": str(target.date()), "model_version": champ["version"], "features": FEATURES,
        "temp_columns": TEMP_COLUMNS, "level_column": LEVEL, "rows": feats[FEATURES].round(4).to_numpy().tolist(),
        "temps": [round(v, 2) for v in temps], "prev23": [round(float(v), 2) for v in prev23.to_numpy()],
        "spec": spec, "max_shift": MAX_SHIFT,
        "trained_temp_range": [round(float(seen.min()), 1), round(float(seen.max()), 1)],
        "forecast": {c: fc[c].round(1).tolist() for c in ("p10", "p50", "p90")},
        "hour_ending": [int(h) for h in fc["hour_ending"]],
    }  # fmt: skip
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "whatif.json").write_text(json.dumps(data))
    return data


def copy_runtime(vendor: Path, site: Path) -> bool:
    if not all((vendor / f).exists() for f in RUNTIME):
        log.warning("ONNX Runtime Web not found in %s; the what-if page will show a notice", vendor)
        return False
    (site / "vendor").mkdir(parents=True, exist_ok=True)
    for f in RUNTIME:
        shutil.copy(vendor / f, site / "vendor" / f)
    return True


# --------------------------------------------------------------------------- page
PAGE_CSS = """
.ctrl { display:flex; flex-wrap:wrap; gap:14px 22px; align-items:center; margin:4px 0 14px; }
.ctrl label { font-weight:600; }
.ctrl input[type=range] { width:min(420px, 100%); accent-color:#104281; }
.delta { font-size:26px; font-weight:650; min-width:92px; font-variant-numeric:tabular-nums; }
.presets button { border:1px solid #c3c2b7; background:#fff; border-radius:999px; padding:3px 11px; font-size:13px;
  color:#104281; cursor:pointer; margin:0 6px 6px 0; }
#status { font-size:13px; color:#52514e; } #status.err { color:#c4352b; }
.warn { font-size:13px; color:#8a5a00; background:#fbf0d9; border-radius:8px; padding:6px 10px; display:none; }
"""

PAGE_JS = """
const $ = (id) => document.getElementById(id);
const status = (t, err = false) => { $("status").textContent = t; $("status").className = err ? "err" : ""; };
const data = await (await fetch("data/whatif.json")).json();
const col = Object.fromEntries(data.features.map((f, i) => [f, i]));
let sessions = null;
// Same formulas as features.build_features (see whatif.shift in the repository, which the tests check)
function shifted(delta) {
  const t = data.temps.map((v) => v + delta);
  const win = [...data.prev23, ...t];
  const mean = t.reduce((a, b) => a + b, 0) / 24, max = Math.max(...t);
  return data.rows.map((r, h) => {
    const x = r.slice();
    x[col.temp] = t[h];
    x[col.heating_degrees] = Math.max(15.5 - t[h], 0);
    x[col.cooling_degrees] = Math.max(t[h] - 18, 0);
    x[col.temp_roll24] = win.slice(h, h + 24).reduce((a, b) => a + b, 0) / 24;
    x[col.temp_day_mean] = mean;
    x[col.temp_day_max] = max;
    return x;
  });
}
async function predict(delta) {
  const rows = shifted(delta);
  const X = new ort.Tensor("float32", Float32Array.from(rows.flat()), [24, data.features.length]);
  const out = {};
  for (const [q, s] of Object.entries(sessions)) out[q] = Array.from((await s.run({ features: X }))[s.outputNames[0]].data);
  // model output -> MW, as features.to_demand: ratio x recent level, conformal widening, quantiles sorted
  const ratio = data.spec.target === "ratio_to_recent_level", adj = data.spec.interval_adjust_mw || 0;
  return rows.map((r, h) => {
    const base = ratio ? r[col[data.level_column]] : 1;
    return [out.p10[h] * base - adj, out.p50[h] * base, out.p90[h] * base + adj].sort((a, b) => a - b);
  });
}
const mw = (v) => Math.round(v).toLocaleString() + " MW";
const signed = (v, unit) => (v >= 0 ? "+" : "−") + Math.abs(Math.round(v)).toLocaleString() + unit;
const x = data.hour_ending.map((h) => h + ":00");
let base = null;
async function update() {
  const delta = parseFloat($("delta").value);
  $("deltaText").textContent = (delta > 0 ? "+" : delta < 0 ? "−" : "±") + Math.abs(delta).toFixed(1) + " °C";
  const p = await predict(delta);
  const p50 = p.map((r) => r[1]), peak = Math.max(...p50), at = p50.indexOf(peak);
  const bPeak = Math.max(...base.map((r) => r[1]));
  const energy = p50.reduce((a, b) => a + b, 0), bEnergy = base.reduce((a, r) => a + r[1], 0);
  const tmean = data.temps.reduce((a, b) => a + b, 0) / 24 + delta;
  $("peak").textContent = mw(peak);
  $("peakNote").textContent = `hour ending ${data.hour_ending[at]}:00 EST · ${signed(peak - bPeak, " MW")} vs. forecast`;
  $("energy").textContent = (energy / 1000).toFixed(1) + " GWh";
  $("energyNote").textContent = `${signed(energy - bEnergy, " MWh")} (${((100 * (energy - bEnergy)) / bEnergy).toFixed(1).replace("-", "−")}%) over the day`;
  $("temp").textContent = tmean.toFixed(1) + " °C";
  $("tempNote").textContent = `daily mean; forecast ${(tmean - delta).toFixed(1)} °C`;
  $("band").textContent = Math.round(p[at][0]).toLocaleString() + "–" + mw(p[at][2]);
  const t = data.temps.map((v) => v + delta), [lo, hi] = data.trained_temp_range;
  $("range").style.display = Math.min(...t) < lo || Math.max(...t) > hi ? "block" : "none";
  const color = delta > 0 ? "#eb6834" : delta < 0 ? "#2a78d6" : "#104281";
  const traces = [
    { x, y: base.map((r) => r[2]), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: base.map((r) => r[0]), mode: "lines", fill: "tonexty", fillcolor: "rgba(137,135,129,0.18)", line: { width: 0 },
      name: "Forecast 80% interval", hoverinfo: "skip" },
    { x, y: base.map((r) => r[1]), mode: "lines", name: "Forecast (actual weather forecast)", line: { color: "#898781", width: 2, dash: "dot" },
      hovertemplate: "%{x}: %{y:,.0f} MW<extra>forecast</extra>" },
    { x, y: p.map((r) => r[2]), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: p.map((r) => r[0]), mode: "lines", fill: "tonexty", fillcolor: color + "22", line: { width: 0 },
      name: "Scenario 80% interval", hoverinfo: "skip" },
    { x, y: p50, mode: "lines", name: `Scenario (${$("deltaText").textContent})`, line: { color, width: 2.5 },
      hovertemplate: "%{x}: %{y:,.0f} MW<extra>scenario</extra>" },
  ];
  Plotly.react("chart", traces, {
    height: 420, margin: { l: 64, r: 16, t: 20, b: 90 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: "#52514e", size: 12 },
    xaxis: { title: { text: "Hour ending (EST)" }, gridcolor: "#e1e0d9" },
    yaxis: { title: { text: "MW" }, gridcolor: "#e1e0d9", tickformat: ",.0f" },
    legend: { orientation: "h", y: -0.25, x: 0 }, hovermode: "x unified",
  }, { displayModeBar: false, responsive: true });
}
try {
  if (!RUNTIME) throw new Error("ONNX Runtime Web is not bundled in this build.");
  globalThis.ort = await import("./vendor/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("./vendor/", location.href).href;
  ort.env.wasm.numThreads = 1;
  status("Loading the model…");
  sessions = Object.fromEntries(await Promise.all(["p10", "p50", "p90"].map(async (q) =>
    [q, await ort.InferenceSession.create(`model/${q}.onnx`)])));
  base = await predict(0);
  const drift = Math.max(...base.map((r, h) => Math.abs(r[1] - data.forecast.p50[h])));
  status(`Model v${data.model_version} running in your browser (matches the published forecast to ${drift.toFixed(1)} MW).`);
  $("delta").disabled = false;
  $("delta").addEventListener("input", update);
  document.querySelectorAll("[data-delta]").forEach((b) => b.onclick = () => { $("delta").value = b.dataset.delta; update(); });
  await update();
} catch (e) { status("Could not run the model here: " + e.message, true); }
"""


def write_page(
    site: Path, data: dict, runtime_ok: bool, repo_url: str, synthetic: bool, plotly_version: str
) -> None:
    from .report_base import CSS

    day = pd.Timestamp(data["target_day"])
    lo, hi = data["trained_temp_range"]
    m = data["max_shift"]
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: built by the test suite; these numbers are not real</div>" if synthetic else ""  # fmt: skip
    presets = "".join(f"<button type='button' data-delta='{d}'>{label}</button>"
                      for d, label in [(-10, "Cold snap (−10 °C)"), (-5, "−5 °C"), (0, "Forecast"), (5, "+5 °C"),
                                       (10, "Heat wave (+10 °C)")])  # fmt: skip
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ontario Demand Weather What-If</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
<script src="https://cdn.plot.ly/plotly-{plotly_version}.min.js" charset="utf-8"></script>
<style>{CSS}{PAGE_CSS}</style></head><body>{banner}
<header><div class="wrap"><h1>What if tomorrow is warmer or colder?</h1>
<p>Shift the weather forecast for {day:%A, %B %-d, %Y} and watch Ontario's forecast demand respond. The champion
model (v{data["model_version"]}) runs inside your browser with ONNX Runtime Web, so every position of the slider is
a real model prediction, not a precomputed curve.</p>
<a class="btn" href="./">Back to the dashboard</a><a class="btn" href="api.html">API reference</a>
<a class="btn" href="{repo_url}">Code on GitHub</a></div></header>
<main><section>
<div class="ctrl"><label for="delta">Temperature change, every hour of the day</label>
<input id="delta" type="range" min="-{m}" max="{m}" step="0.5" value="0" disabled>
<span class="delta" id="deltaText">±0.0 °C</span></div>
<div class="presets">{presets}</div>
<p><span id="status">Loading…</span></p>
<div class="cards">
<div class="card"><div class="label">Peak demand</div><div class="value" id="peak">–</div><div class="note" id="peakNote"></div></div>
<div class="card"><div class="label">Peak hour 80% interval</div><div class="value" id="band" style="font-size:20px">–</div><div class="note">10th to 90th percentile</div></div>
<div class="card"><div class="label">Energy over the day</div><div class="value" id="energy">–</div><div class="note" id="energyNote"></div></div>
<div class="card"><div class="label">Temperature</div><div class="value" id="temp">–</div><div class="note" id="tempNote"></div></div>
</div>
<p class="warn" id="range">Part of this scenario is outside the temperatures in the training data ({lo:g} to {hi:g} °C).
Tree models don't extrapolate, so beyond that range demand stops responding.</p>
<div id="chart"></div>
<p class="muted">Only the temperatures change: the demand history and the calendar (weekday, holidays) stay as they
are. The temperature is a population-weighted average of Toronto, Ottawa and London. The 80% interval comes from the
10th and 90th percentile models, with the same conformal widening as the published forecast. In winter, colder means
more electric heating; in summer, hotter means more air conditioning.</p>
</section></main>
<footer>Data: IESO public reports · weather by Open-Meteo (CC BY 4.0) · <a href="{repo_url}">Source code</a></footer>
<script type="module">
const RUNTIME = {"true" if runtime_ok else "false"};
{PAGE_JS}
</script></body></html>"""
    (site / "whatif.html").write_text(page, encoding="utf-8")
