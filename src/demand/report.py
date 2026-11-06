"""Dashboard (site/index.html), API reference (site/api.html + openapi.json) and results.json."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from . import registry
from .config import ROOT, Config
from .features import FEATURES
from .report_base import BLUE, CSS, LAYOUT_JS, MUTED, ORANGE, VIOLET, _card, _div, _int, _style, _table

LABELS = {
    "hour": "Hour of day", "day_of_week": "Day of week", "is_weekend": "Weekend", "is_holiday": "Holiday",
    "doy_sin": "Season (sin)", "doy_cos": "Season (cos)", "temp": "Temperature", "heating_degrees": "Heating degrees",
    "cooling_degrees": "Cooling degrees", "temp_roll24": "Temperature, last 24 h", "temp_day_mean": "Day's mean temperature",
    "temp_day_max": "Day's max temperature", "temp_prev_day_mean": "Yesterday's mean temperature",
    "lag24": "Demand, same hour yesterday", "lag168": "Demand, same hour last week",
    "same_hour_mean_7d": "Demand, same hour 7-day mean", "prev_day_mean": "Yesterday's mean demand",
    "prev_day_max": "Yesterday's peak demand", "prev_day_min": "Yesterday's minimum demand",
}  # fmt: skip
EXTRA_CSS = """
.cards { grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); }
main a { color:#2a78d6; }
td.l:first-child { min-width:120px; }
pre { background:#f0efec; border-radius:8px; padding:12px; overflow-x:auto; font-size:12.5px; }
.arch { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 14px; }
.arch .box { flex:1 1 160px; background:#f4f3ef; border-radius:10px; padding:10px 12px; font-size:13px; color:#52514e; }
.arch .box b { display:block; color:#0b0b0b; margin-bottom:2px; }
"""


def _p(v, d=2) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.{d}f}%"


def _mw(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:,.0f} MW"


# --------------------------------------------------------------------------- figures
def fig_forecast(frame: pd.DataFrame, fc: pd.DataFrame, scored: pd.DataFrame) -> go.Figure:
    start = pd.Timestamp(fc["ts"].min()) - pd.Timedelta(days=10)
    actual = frame.loc[
        (frame.index >= start) & (frame.index < pd.Timestamp(fc["ts"].min())), "demand_mw"
    ].dropna()
    past = scored[scored["ts"] >= start] if len(scored) else scored
    fig = go.Figure()
    days = (
        [(d, MUTED, "Issued forecasts (scored)") for _, d in past.groupby("target_day")] if len(past) else []
    )
    days.append((fc, BLUE, "Next-day forecast"))
    shown = set()
    for d, color, name in days:  # one trace per day so days without a forecast stay as gaps
        fill = "rgba(42,120,214,0.15)" if color == BLUE else "rgba(137,135,129,0.18)"
        fig.add_trace(
            go.Scatter(
                x=pd.concat([d["ts"], d["ts"][::-1]]),
                y=pd.concat([d["p90"], d["p10"][::-1]]),
                fill="toself",
                fillcolor=fill,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(go.Scatter(x=d["ts"], y=d["p50"], mode="lines", name=name, showlegend=name not in shown,
                                 legendgroup=name, line=dict(color=color, width=2, dash="dot" if color == MUTED else "solid"),
                                 hovertemplate="%{x|%b %d %H:00}: %{y:,.0f} MW"))  # fmt: skip
        shown.add(name)
    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode="lines", name="Actual demand", line=dict(color=ORANGE, width=2),
                             hovertemplate="%{x|%b %d %H:00}: %{y:,.0f} MW"))  # fmt: skip
    _style(fig, "Ontario demand: actual, past forecasts and the next-day forecast (shaded: 80% interval)", height=420,
           y_title="MW")  # fmt: skip
    return fig


def fig_live(daily: pd.DataFrame) -> go.Figure:
    d = daily.sort_values("target_day").tail(60)
    fig = go.Figure(go.Bar(x=pd.to_datetime(d["target_day"]), y=d["mape"], marker_color=BLUE,
                           customdata=d["model_version"],
                           hovertemplate="%{x|%b %d}: %{y:.2f}% (model v%{customdata})<extra></extra>"))  # fmt: skip
    fig.update_traces(marker_line_width=0)
    _style(fig, "Live error of each day's issued forecast (MAPE)", height=320, hovermode="closest", showlegend=False,
           y_title="Mean absolute % error")  # fmt: skip
    return fig


def fig_backtest(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, color in (
        ("LightGBM (quantile)", BLUE),
        ("Same hour last week", MUTED),
        ("Same hour yesterday", VIOLET),
    ):
        d = bt[bt["model"] == name]
        fig.add_trace(go.Scatter(x=d["month"], y=d["mape"], mode="lines+markers", name=name, line=dict(color=color, width=2),
                                 marker=dict(size=7), hovertemplate="%{x}: %{y:.2f}%"))  # fmt: skip
    _style(fig, "Backtest: monthly MAPE, model vs. naive baselines", height=400, y_title="MAPE (%)",
           legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
           margin=dict(l=64, r=20, t=60, b=100))  # fmt: skip
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(type="category")
    return fig


def fig_importance(imp: pd.DataFrame) -> go.Figure:
    d = imp.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["gain_share"], y=[LABELS.get(f, f) for f in d["feature"]], orientation="h",
                           marker_color=BLUE, hovertemplate="%{y}: %{x:.1%} of gain<extra></extra>"))  # fmt: skip
    fig.update_traces(marker_line_width=0)
    _style(fig, "What the median model relies on (share of split gain)", height=380, hovermode="closest",
           showlegend=False, margin=dict(l=210, r=20, t=74, b=40))  # fmt: skip
    fig.update_xaxes(tickformat=".0%")
    return fig


# --------------------------------------------------------------------------- page
def api_page(cfg: Config) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "serving"))
    from app.main import app

    (cfg.site_dir / "openapi.json").write_text(json.dumps(app.openapi(), indent=2))
    (cfg.site_dir / "api.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Forecast API reference</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'></head><body>"
        "<redoc spec-url='openapi.json'></redoc>"
        "<script src='https://cdn.jsdelivr.net/npm/redoc@2.5.1/bundles/redoc.standalone.js'></script></body></html>"
    )


def build_site(cfg: Config, frame, fc, scored, daily, run: dict) -> dict:
    site = cfg.site_dir
    site.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    champ = run["champion"]
    live = run["live"]
    versions = registry.versions(cfg)
    run_id = registry.champion(cfg)["run_id"]
    import mlflow

    rep = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reports")
    bt = pd.read_csv(f"{rep}/backtest.csv")
    imp = pd.read_csv(f"{rep}/importance/importance.csv")

    peak = fc.loc[fc["p50"].idxmax()]
    adjust = float(champ["metrics"].get("interval_adjust_mw", 0.0))
    overall = bt.groupby("model")[["mape", "mae", "peak_ape", "coverage_80"]].mean()
    model_bt, naive_bt = overall.loc["LightGBM (quantile)"], overall.loc["Same hour last week"]
    improvement = 1 - model_bt["mape"] / naive_bt["mape"]
    drift = run["drift"]
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: generated for automated tests; these numbers are not real</div>" if cfg.fixture else ""  # fmt: skip

    cards = "".join([
        _card(f"Peak forecast, {pd.Timestamp(run['target_day']):%b %d}", _mw(peak["p50"]),
              f"hour ending {int(peak['hour_ending'])}:00 EST · 80% interval {peak['p10']:,.0f}–{peak['p90']:,.0f}"),
        _card("Live error (MAPE)", _p(live.get("mape")) if live.get("days") else "–",
              f"issued forecasts, last {live.get('days', 0)} days" if live.get("days") else "starts after the first scored day"),
        _card("Backtest MAPE", _p(model_bt["mape"]), f"{improvement:.0%} lower than same-hour-last-week ({_p(naive_bt['mape'])})"),
        _card("80% interval coverage", _p(live["coverage_80"], 0) if live.get("days", 0) >= 7 else _p(model_bt["coverage_80"], 0),
              f"live, last {live['days']} days; target 80%" if live.get("days", 0) >= 7 else
              f"backtest before calibration; band widened {adjust:+,.0f} MW each side"),
        _card("Champion model", f"v{champ['version']}", f"data through {champ['tags'].get('data_end')}"),
        _card("Input drift", f"{drift['drifted']}/{drift['n_columns']} columns" if drift.get("available") else "–",
              "vs. same weeks last year" if drift.get("available") else drift.get("reason", "")),
    ])  # fmt: skip

    bt_rows = []
    for name, r in overall.sort_values("mape").iterrows():
        bt_rows.append([html.escape(name), _p(r["mape"]), _mw(r["mae"]), _p(r["peak_ape"]),
                        _p(r["coverage_80"], 0) if pd.notna(r["coverage_80"]) else "–"])  # fmt: skip
    reg_rows = []
    for v in versions[:12]:
        mm = v["metrics"]
        status = (
            "★ champion"
            if "champion" in v["aliases"]
            else ("previous" if "previous" in v["aliases"] else v.get("status", ""))
        )
        gate = _p(mm.get("gate_challenger_mape"))
        if mm.get("gate_champion_mape") is not None:
            gate += f" vs {_p(mm.get('gate_champion_mape'))}"
        reg_rows.append([f"<b>v{v['version']}</b><div class='muted'>{v['created']}</div>",
                         f"{html.escape(status)}<div class='muted'>{html.escape(v.get('decision', ''))}</div>",
                         v.get("data_end", "–"), _p(mm.get("backtest_mape")), gate, html.escape(v.get("trigger", ""))])  # fmt: skip
    drift_rows = [[html.escape(LABELS.get(c["column"], c["column"]) if c["column"] != "demand_mw" else "Demand (target)"),
                   html.escape(c["method"]), f"{c['score']:.3g}", f"{c['threshold']:g}",
                   "<span class='chip warn'>drift</span>" if c["drifted"] else "<span class='chip good'>stable</span>"]
                  for c in drift.get("columns", [])]  # fmt: skip
    fc_rows = [[f"{int(r.hour_ending)}:00", f"{r.temp_forecast:.1f} °C", _mw(r.p10), f"<b>{_mw(r.p50)}</b>", _mw(r.p90)]
               for r in fc.itertuples()]  # fmt: skip
    rt = run.get("retrain")
    retrain_line = (
        f"This run retrained ({html.escape(rt['trigger'])}): version {rt['version']} was "
        f"<b>{'promoted' if rt['promoted'] else 'not promoted'}</b>: {html.escape(rt['decision'])}."
        if rt else "No retraining today: the champion is retrained weekly, or early if live error rises."
    )  # fmt: skip
    dq = run["data_checks"]
    api_example = (
        f"docker run -p 8000:8000 {cfg.image}:latest\n"
        "curl -s localhost:8000/forecast/example | curl -s -X POST localhost:8000/forecast \\\n"
        "     -H 'Content-Type: application/json' -d @-"
    )

    api_page(cfg)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ontario Demand Forecast</title>
<meta name="description" content="Next-day hourly forecast of Ontario electricity demand: LightGBM quantile models, MLflow registry, ONNX + FastAPI serving in Docker, Evidently monitoring.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>{CSS}{EXTRA_CSS}</style></head>
<body>{banner}
<header><div class="wrap">
  <h1>Ontario electricity demand: next-day forecast</h1>
  <p>An hourly forecast of tomorrow's Ontario electricity demand with an 80% prediction interval, retrained, checked
  and served automatically: LightGBM quantile models tracked in an MLflow registry, exported to ONNX, served by a
  FastAPI container, and monitored for accuracy and drift. Updated every day from IESO data.</p>
  <a class="btn" href="{cfg.repo_url}">Code on GitHub</a><a class="btn" href="api.html">API reference</a><a class="btn" href="drift.html">Drift report</a>
</div></header>
<main>
<div class="cards">{cards}</div>

<section><h2>Forecast</h2>
<p>Each morning (EST) the champion model forecasts the next day once the previous day's demand is complete, using
the weather forecast for that day. Grey shows earlier forecasts as they were issued; orange is what actually happened.</p>
{_div(fig_forecast(frame, fc, scored))}
<details><summary>Hourly forecast for {run["target_day"]}</summary>
{_table(["Hour ending (EST)", "Temperature", "Low (p10)", "Forecast (p50)", "High (p90)"], fc_rows)}</details>
</section>

<section><h2>How accurate is it?</h2>
<p><b>Live:</b> every issued forecast is scored once the day is over, which is the honest number. <b>Backtest:</b> for
each of the last {bt["month"].nunique()} months the model is retrained on everything before it and forecasts every day of the month
using observed temperatures, a best case for the weather input. Both are compared with naive baselines.</p>
<div class="grid2">{_div(fig_live(daily)) if len(daily) else "<p class='muted'>Live scores appear once the first forecast day has finished.</p>"}
<div><h3>Backtest summary</h3>{_table(["Model", "MAPE", "MAE", "Daily peak error", "80% coverage*"], bt_rows)}
<p class="muted">*Before calibration. The champion's 80% band is then widened by {adjust:+,.0f} MW on each side
(conformalized quantile regression on these backtest errors) so it should contain the truth about 80% of the time;
live coverage shows whether it does.</p></div></div>
<div class="grid2">{_div(fig_backtest(bt))}{_div(fig_importance(imp))}</div>
</section>

<section><h2>Model registry (MLflow)</h2>
<p>{retrain_line} A retrained model is promoted only if it beats the seasonal-naive baseline in the backtest and,
on the most recent days that neither model was trained on (up to {cfg.gate_days}), its error is no worse than the
champion's (within {cfg.gate_tolerance - 1:.0%}). Every exported model must match LightGBM to within 1 MW in ONNX.</p>
{_table(["Version", "Status and decision", "Data through", "Backtest MAPE", "Gate: new vs champion", "Trigger"], reg_rows, left_cols=2)}
</section>

<section><h2>Monitoring</h2>
<p>Input drift (Evidently): the last {cfg.drift_days} days against the same weeks a year earlier, so normal seasonal
change is not flagged. Live error is also watched: if the last week's error runs {cfg.live_alert_ratio:g}× the backtest
error, the model is retrained early. Data checks this run: {_int(dq["rows"])} hourly rows,
{dq["missing_hours"]} missing hours, {dq["out_of_range"]} out-of-range values removed.</p>
{_table(["Column", "Test", "Score", "Threshold", "Status"], drift_rows) if drift_rows else "<p class='muted'>" + html.escape(drift.get("reason", "")) + "</p>"}
<p><a href="drift.html">Full Evidently report →</a></p>
</section>

<section><h2>Serving</h2>
<div class="arch">
<div class="box"><b>Train</b>LightGBM quantile models (p10, p50, p90), {len(FEATURES)} features</div>
<div class="box"><b>Export</b>ONNX, parity-checked against LightGBM</div>
<div class="box"><b>Register</b>MLflow run + model version, alias <code>champion</code></div>
<div class="box"><b>Package</b>FastAPI + onnxruntime in a Docker image on GitHub Container Registry</div>
</div>
<p>The image is rebuilt and pushed whenever a new champion is promoted, and smoke-tested before every push. Run it
anywhere Docker runs (the example request is built from real recent data):</p>
<pre>{html.escape(api_example)}</pre>
<p><a href="api.html">API reference (OpenAPI)</a> · endpoints: <code>GET /health</code>, <code>GET /model</code>,
<code>GET /forecast/example</code>, <code>POST /forecast</code>.</p>
</section>

<section><h2>How it works</h2><ul class="method">
<li><b>Data:</b> IESO hourly Ontario demand (public reports, EST all year) and Open-Meteo hourly temperatures for
Toronto, Ottawa and London weighted by population (reanalysis for history, forecast for the next day).</li>
<li><b>Features:</b> calendar and Ontario holidays, temperature with heating and cooling degrees, and demand lags that
only use data up to the end of the previous day. The same feature code runs in training and in the API.</li>
<li><b>Automation:</b> GitHub Actions runs the pipeline daily: ingest → score past forecasts → retrain and gate
(weekly) → forecast → drift report → publish; MLflow state and the forecast log are kept between runs.</li>
<li><b>Limits:</b> one temperature proxy for the whole province; no special handling of extreme events or
large industrial outages; the backtest uses observed temperatures, so live error is expected to be somewhat higher.</li>
</ul></section>
</main>
<footer>Data: IESO public reports · Open-Meteo (CC BY 4.0) · Built {built_at} · <a href="{cfg.repo_url}">Source code</a></footer>
<script>{LAYOUT_JS}</script>
</body></html>"""
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").touch()

    summary = {
        "target_day": run["target_day"], "data_end": run["data_end"], "peak": {k: float(peak[k]) for k in ("p10", "p50", "p90")},
        "peak_hour_ending": int(peak["hour_ending"]), "live": live,
        "backtest": {"model": model_bt.to_dict(), "naive_week": naive_bt.to_dict(),
                     "naive_day": overall.loc["Same hour yesterday"].to_dict(), "months": int(bt["month"].nunique()),
                     "improvement_vs_naive": float(improvement)},
        "champion": {"version": champ["version"], "data_end": champ["tags"].get("data_end")},
        "versions": len(versions), "retrain": {k: rt[k] for k in ("version", "promoted", "decision", "trigger")} if rt else None,
        "drift": {k: drift.get(k) for k in ("available", "drifted", "n_columns", "drift_share")},
        "image": cfg.image, "synthetic": cfg.fixture, "built_at": built_at,
    }  # fmt: skip
    (site / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
