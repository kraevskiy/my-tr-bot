"""Deep analytics for backtest / experiment results -> one self-contained HTML report.

Sections
  1. Headline: trades, avgR, t, bootstrap 95% CI, p-value, p adjusted for the number of
     variants tested (Bonferroni and an effective-N from correlation of variant results)
  2. Equity curve and drawdown (underwater) in R
  3. Distribution of R, MFE vs MAE (how far trades go for / against us)
  4. Breakdowns: symbol, killzone, setup, side, liquidity tier, weekday, hour, month,
     stop size quartile, exit kind, year/quarter
  5. Monte Carlo: drawdown distribution, risk-per-trade table vs prop-firm limits
  6. Walk-forward of the SELECTION process across all variants in the run folder
  7. Candidate log (if present): value of each filter measured on market-entry outcomes
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smcbot.config import NY_TZ

SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"      # validated categorical slots 1-2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'<img alt="chart" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load_trades(path: str | Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    f = t[t["status"] == "filled"].copy()
    f["ts"] = pd.to_datetime(f["signal_time"], utc=True).dt.tz_convert(NY_TZ)
    return f.sort_values("ts").reset_index(drop=True)


def headline(r: np.ndarray, n_variants: int = 1, n_eff: float | None = None, seed=0) -> dict:
    from scipy import stats
    n = len(r)
    if n < 3:
        return {"trades": n}
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1)) if sd > 0 else np.nan
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(r, n).mean() for _ in range(10000)])
    wins, losses = r[r > 0.05], r[r < -0.05]
    out = {"trades": n, "avgR": r.mean(), "totalR": r.sum(), "win%": 100 * len(wins) / n,
           "PF": wins.sum() / -losses.sum() if len(losses) else np.nan, "t": t, "p": p,
           "CI95_low": np.percentile(boot, 2.5), "CI95_high": np.percentile(boot, 97.5),
           "P(avgR<=0)": (boot <= 0).mean()}
    if n_variants > 1:
        out["p_bonferroni"] = min(1.0, p * n_variants)
    if n_eff:
        out["n_eff_variants"] = n_eff
        out["p_adj_eff"] = min(1.0, p * n_eff)
    return out


def monte_carlo(r: np.ndarray, years: float, risks=(0.3, 0.5, 1.0), sims=10000, seed=1,
                daily_limit=5.0, total_limit=10.0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dd, fin = np.empty(sims), np.empty(sims)
    for k in range(sims):
        s = rng.choice(r, len(r))
        eq = s.cumsum()
        dd[k] = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
        fin[k] = eq[-1]
    rows = []
    for risk in risks:
        rows.append({"risk %/trade": risk,
                     "median maxDD %": np.median(dd) * risk,
                     "95% maxDD %": np.percentile(dd, 95) * risk,
                     f"P(DD > {total_limit:g}%)": (dd * risk > total_limit).mean(),
                     "median return %/year": np.median(fin) * risk / years,
                     "5% worst return %/year": np.percentile(fin, 5) * risk / years})
    return {"dd_median_R": np.median(dd), "dd_95_R": np.percentile(dd, 95),
            "dd_99_R": np.percentile(dd, 99)}, pd.DataFrame(rows)


def breakdowns(f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    g = f.copy()
    g["weekday"] = g["ts"].dt.day_name().str[:3]
    g["hour_NY"] = g["ts"].dt.hour
    g["month"] = g["ts"].dt.month
    g["quarter"] = g["ts"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    g["setup_base"] = g["setup"].astype(str).str.split("+").str[0]
    if "risk_pips" in g and g["risk_pips"].notna().sum() > 8:
        g["stop_quartile"] = pd.qcut(g["risk_pips"], 4, duplicates="drop").astype(str)
    # news context (trades from newer runs carry news_* columns; older files: annotate here)
    if "news_bucket" not in g:
        try:
            from smcbot.market.news import annotate, load_default
            cal = load_default()
            if cal is not None and "symbol" in g:
                tmp = g.copy()
                tmp["signal_time"] = tmp["ts"]
                g = annotate(tmp, cal)
        except Exception:      # analytics must never fail on news
            pass
    if "news_day" in g:
        g["news_tier1_day"] = np.where(g["news_day"].fillna("").astype(str) != "", "yes", "no")
        g["news_main"] = g["news_day"].fillna("").astype(str).str.split("+").str[0].replace("", "none")
    if "holiday" in g:
        g["holiday"] = g["holiday"].fillna("").astype(str).replace("", "none")
    out = {}
    for col in ("symbol", "killzone", "setup_base", "side", "tier", "weekday", "hour_NY", "month",
                "stop_quartile", "exit_kind", "quarter", "smt", "news_bucket", "news_tier1_day", "news_main", "holiday", "grade"):
        if col in g:
            out[col] = g.groupby(col)["R_result"].agg(trades="count", avgR="mean", totalR="sum").round(2)
    return out


def variant_matrix(run_dir: Path, mode: str) -> pd.DataFrame:
    """Quarterly totalR of every variant in an experiments folder (rows=quarters)."""
    cols = {}
    for p in sorted(run_dir.glob(f"*/trades_{mode}.csv")):
        f = load_trades(p)
        if len(f) >= 10:
            cols[p.parent.name] = f.groupby(f["ts"].dt.tz_localize(None).dt.to_period("Q"))["R_result"].sum()
    return pd.DataFrame(cols).fillna(0.0).sort_index()


def effective_n(q: pd.DataFrame) -> float | None:
    if q.shape[1] < 2:
        return None
    c = np.corrcoef(q.T.values)
    c = np.nan_to_num(c, nan=0.0)
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 1e-9]
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def walk_forward_selection(q: pd.DataFrame, fixed: str, lookback=4) -> pd.DataFrame:
    rows = []
    idx = list(q.index)
    for i in range(lookback, len(idx)):
        trail = q.loc[idx[i - lookback:i]].sum()
        pick = trail.idxmax()
        rows.append({"quarter": str(idx[i]), "picked_by_trailing": pick, "picked_R": q.loc[idx[i], pick],
                     "median_variant_R": q.loc[idx[i]].median(), f"{fixed}_R": q.loc[idx[i], fixed]
                     if fixed in q else np.nan})
    return pd.DataFrame(rows)


def candidate_filter_value(cands: pd.DataFrame) -> pd.DataFrame | None:
    if cands is None or "fwd_R2" not in cands:
        return None
    v = cands[cands["fwd_R2"].notna()].copy()
    v["rl"] = v["reasons"].fillna("").map(lambda s: s.split(",") if s else [])
    rows = [{"filter": "ALL PASSED (signals)", "n": int((v["rl"].map(len) == 0).sum()),
             "fwd_R2": v.loc[v["rl"].map(len) == 0, "fwd_R2"].mean()}]
    for r in sorted({x for l in v["rl"] for x in l}):
        only = v[v["rl"].map(lambda l: l == [r])]
        if len(only):
            rows.append({"filter": f"rejected ONLY by {r}", "n": len(only), "fwd_R2": only["fwd_R2"].mean(),
                         "se": only["fwd_R2"].std(ddof=1) / np.sqrt(len(only)) if len(only) > 1 else np.nan})
    return pd.DataFrame(rows).round(3)


def _charts(f: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    eq = f["R_result"].cumsum()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    a1.plot(f["ts"].dt.tz_localize(None), eq, color=SERIES_1, linewidth=2)
    _style(a1, "Equity, R (cumulative)")
    under = eq - eq.cummax()
    a2.fill_between(f["ts"].dt.tz_localize(None), under, 0, color=SERIES_2, alpha=0.35, linewidth=0)
    a2.plot(f["ts"].dt.tz_localize(None), under, color=SERIES_2, linewidth=1.2)
    _style(a2, "Drawdown, R")
    out.append(_fig_to_img(fig))

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(9, 3.4))
    b1.hist(f["R_result"], bins=np.arange(-1.25, 3.5, 0.25), color=SERIES_1, edgecolor="white", linewidth=2)
    _style(b1, "Result per trade, R")
    if "mfe_R" in f and "mae_R" in f:
        w = f["R_result"] > 0
        b2.scatter(f.loc[w, "mae_R"], f.loc[w, "mfe_R"], s=28, color=SERIES_1, edgecolor="white", linewidth=1, label="winners")
        b2.scatter(f.loc[~w, "mae_R"], f.loc[~w, "mfe_R"], s=28, color=SERIES_2, edgecolor="white", linewidth=1, label="losers / BE")
        b2.set_xlabel("MAE: worst move against, R", fontsize=8, color=INK_2)
        b2.set_ylabel("MFE: best move for us, R", fontsize=8, color=INK_2)
        b2.legend(frameon=False, fontsize=8)
        _style(b2, "How far trades went (MFE vs MAE)")
    out.append(_fig_to_img(fig))
    return out


def _tbl(df: pd.DataFrame, index=True) -> str:
    return df.to_html(index=index, float_format=lambda x: (f"{x:.4f}" if abs(x) < 0.01 and x != 0 else f"{x:,.2f}"), border=0, classes="t")


def build_report(trades_path: str | Path, out_html: str | Path, title: str = "",
                 run_dir: str | Path | None = None, mode: str = "b",
                 candidates_path: str | Path | None = None) -> dict:
    f = load_trades(trades_path)
    r = f["R_result"].to_numpy()
    years = max((f["ts"].max() - f["ts"].min()).days / 365.25, 0.25) if len(f) else 1
    q, n_eff, n_var, wf = None, None, 1, None
    if run_dir:
        q = variant_matrix(Path(run_dir), mode)
        n_var = q.shape[1]
        n_eff = effective_n(q)
        wf = walk_forward_selection(q, Path(trades_path).parent.name)
    head = headline(r, n_var, n_eff)
    r_mc = r
    if "R_w" in f and "risk_mult" in f and (f["risk_mult"] != 1.0).any():     # v4: risk by setup grade
        w = f["R_w"].to_numpy()
        eq = np.cumsum(w)
        head.update({"risk-weighted totalR": float(w.sum()),
                     "risk-weighted maxDD_R": float((np.maximum.accumulate(eq) - eq).max()),
                     "grade A trades / avgR": f"{(f['grade'] == 'A').sum()} / {f.loc[f['grade'] == 'A', 'R_result'].mean():.2f}",
                     "grade B trades / avgR": f"{(f['grade'] == 'B').sum()} / {f.loc[f['grade'] == 'B', 'R_result'].mean():.2f}"})
        r_mc = w                                                              # Monte Carlo on what the account sees
    mc, mc_tbl = monte_carlo(r_mc, years) if len(r) > 5 else ({}, pd.DataFrame())
    br = breakdowns(f) if len(f) else {}
    cands = None
    if candidates_path and Path(candidates_path).exists():
        cands = candidate_filter_value(pd.read_csv(candidates_path))
    extra = {}
    if "mfe_R" in f and len(f):
        losers = f[f["R_result"] < 0]
        winners = f[f["R_result"] > 0]
        extra = {"winners: avg MAE R": winners["mae_R"].mean(), "losers: avg MFE R": losers["mfe_R"].mean(),
                 "losers that reached +0.5R": (losers["mfe_R"] >= 0.5).mean(),
                 "e-ratio (avg MFE / avg MAE)": f["mfe_R"].mean() / max(f["mae_R"].mean(), 1e-9),
                 "median wait for fill, min": f.get("fill_wait_min", pd.Series(dtype=float)).median(),
                 "median time in trade, min": f.get("duration_min", pd.Series(dtype=float)).median(),
                 "trades per year": len(f) / years}

    css = """body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:980px;margin:24px auto;padding:0 16px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #e6e5e1;padding-bottom:4px}
table.t{border-collapse:collapse;font-size:12.5px;margin:6px 0 14px}table.t th,table.t td{padding:3px 10px;border-bottom:1px solid #eee;text-align:right}
table.t th{color:#52514e;font-weight:600}.note{color:#52514e;font-size:12.5px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 24px}
img{max-width:100%}"""
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>",
         f"<h1>{title or 'Strategy analysis'}</h1>",
         f"<p class='note'>{len(f)} trades, {f['ts'].min():%Y-%m-%d} → {f['ts'].max():%Y-%m-%d}. Results in R (1R = planned risk).</p>" if len(f) else ""]
    h += ["<h2>1. Is there an edge?</h2>", _tbl(pd.Series(head).to_frame("value"))]
    h.append("<p class='note'>t &gt; 2 and a 95% interval above 0 mean the average is unlikely to be luck for THIS variant. "
             "p_bonferroni / p_adj_eff correct for having tried many variants (effective N accounts for variants being similar). "
             "Values below 0.05 survive that correction.</p>")
    if len(f):
        h += ["<h2>2. Equity and drawdown</h2>"] + _charts(f)[:1]
        h += ["<h2>3. Trade anatomy</h2>"] + _charts(f)[1:]
        if extra:
            h.append(_tbl(pd.Series(extra).to_frame("value")))
    if len(mc_tbl):
        h += ["<h2>4. Monte Carlo (10 000 reshuffles of the trades)</h2>",
              _tbl(pd.Series(mc).to_frame("R")), _tbl(mc_tbl, index=False),
              "<p class='note'>Prop-firm style limit: 10% total drawdown. Daily limits are not modelled here; "
              "the bot already stops the day at −1.5% or 5 SL.</p>"]
    if br:
        h.append("<h2>5. Where the result comes from</h2><div class='grid'>")
        for k, v in br.items():
            h.append(f"<div><b>{k}</b>{_tbl(v)}</div>")
        h.append("</div>")
    if wf is not None and len(wf):
        h += ["<h2>6. Does re-optimising help? (walk-forward of the selection)</h2>",
              "<p class='note'>Each quarter, pick the variant with the best result over the previous 4 quarters and trade it "
              "for the next quarter. If this beats the fixed strategy, re-optimising works; if not, freeze the rules.</p>",
              _tbl(wf.round(2), index=False),
              f"<p>Sum: picked = {wf['picked_R'].sum():.1f}R, median variant = {wf['median_variant_R'].sum():.1f}R, "
              f"fixed = {wf.iloc[:, -1].sum():.1f}R.</p>"]
    if cands is not None:
        h += ["<h2>7. Filter value on the candidate log</h2>",
              "<p class='note'>fwd_R2 = market entry at the signal, +2R target / −1R stop within 4h. "
              "If candidates rejected only by a filter do worse than signals, the filter helps.</p>", _tbl(cands, index=False)]
    h.append("</body></html>")
    Path(out_html).write_text("\n".join(h), encoding="utf-8")
    return {"headline": head, "monte_carlo": mc}
