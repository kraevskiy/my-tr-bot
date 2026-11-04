# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

The only project is `bot/`: an ICT/SMC backtesting bot (`smcbot` package, Python >= 3.11) for EURUSD/GBPUSD. All commands below run from `bot/`.

User-facing docs (`README.md`, `STRATEGY.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `tradingview/README.md`) are written in Ukrainian; code, docstrings and comments are in English. Keep that split.

## Commands

System `python3` on this Mac is 3.9 and too old. Use Docker, or an explicit interpreter such as `/opt/homebrew/bin/python3.12`.

```bash
# Docker — the intended path on Mac/Linux (.env is injected via docker compose env_file)
docker compose build
docker compose run --rm backtest
docker compose run --rm experiments --only V4,V5 --modes c
docker compose run --rm analyze /app/results/<run_id> V5 --mode c
docker compose run --rm tests                       # pytest -q tests inside the image

# Local (no Docker)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt    # or: pip install -e ".[dev]"
.venv/bin/pytest -q                                 # whole suite (97 tests, ~10 s): synthetic data, no network, no market data needed
.venv/bin/pytest tests/test_core.py::test_no_lookahead_in_signals        # single test
.venv/bin/python -m smcbot backtest --synthetic     # smoke-test the pipeline on random data (results mean nothing)

# Research round (real data, env exported as below): ~35 s detector build + 3-6 s per variant
.venv/bin/python -m smcbot experiments --file experiments_r11.json          # one round = one experiments_r<N>.json
.venv/bin/python -m smcbot neighbours --n 60 --out results/nb60.json        # +-20% random neighbours of a preset
.venv/bin/python -m smcbot experiments --file results/nb60.json
.venv/bin/python -m smcbot neighbours --summarize results/<run_id>          # pre-registered robustness verdicts
.venv/bin/python -m smcbot prop results/<run_id>/<variant>/trades_c.csv --risk 0.5,0.75,1,1.5   # prop-firm challenge pass/fail/time simulator
```

**`.env` is not auto-loaded outside Docker/Windows.** There is no python-dotenv; only Docker Compose (`env_file`) and `run_windows.bat` inject it. A bare `python -m smcbot ...` sees real environment variables only and falls back to Docker paths (`DATA_DIR=/app/data`, `RESULTS_DIR=/app/results`), which do not exist on a Mac. For a local run with real data:

```bash
set -a; source .env; set +a
export DATA_DIR=data RESULTS_DIR=results CSV_EURUSD='data/EURUSD_M1*.csv' CSV_GBPUSD='data/GBPUSD_M1*.csv'   # STRATEGY defaults to v6

# Dukascopy (no key needed; cache in data/cache/<SYMBOL>/YYYY-MM.csv.gz; 5 years download in ~10 min; DATE_TO is exclusive)
export DATA_SOURCE=dukascopy DATA_DIR=data RESULTS_DIR=results_dk DATE_FROM=2019-01-01 DATE_TO=2024-01-01
```

Real backtests need MT5 M1 exports in `bot/data/` (`EURUSD_M1*.csv`, `GBPUSD_M1*.csv`); `data/` is empty in a fresh checkout. `results/` is output and dockerignored.

`analyze` and `experiments` default to stop mode `c`, which is the only mode v5/v6 runs produce (`trades_c.csv`), so `--mode c` is no longer needed.

`python -m smcbot config` prints the merged effective config; use it to check what a `.env` + preset combination actually produces.

## Architecture

Strict layering; each layer imports only from the ones above it (`ARCHITECTURE.md` has the module table):

```
config.py / settings.py -> data -> market -> strategy -> engine -> research -> apps + cli
```

Invariants that span several modules. Breaking any of them silently corrupts backtest results:

- **Time and price conventions** (`data/bars.py`): bars are tz-aware `America/New_York`, indexed by OPEN time; prices are BID; the `spread` column is in pips. The forex day starts 17:00 NY, so resampling goes through "trading time" (NY + 7h) to align D1/H4 the way a GMT+2/+3 MT5 server does. Kyiv times in docs are NY + 7 (NY + 6 for a few weeks a year); code always reasons in NY.
- **No look-ahead**: every market object carries `known_at` (swing: `n` bars after it; FVG: after the third candle closes; higher-TF bar: only after its close; Asia pool: 02:00 NY). A signal may only use what is known at its bar's close. `test_no_lookahead_in_signals` and `test_af_no_lookahead_on_synthetic` truncate the future and assert past signals are unchanged.
- **Long-only logic, shorts by mirroring** (`strategy/setups.py`): `SideModel` runs the same code on `mirror(m1)` (prices negated, high/low swapped) and `unmirror` converts results back. Never write a separate short branch.
- **Two-stage signals**: `SideModel.base_candidates()` (stage A) computes every M5 Shift with parameter-independent features once; `signals(cfg)` (stage B) applies a variant's windows, filters, entry, stop, target and grade. Every rejection increments `funnel["x_<reason>"]`. This is why `experiments` runs dozens of variants in minutes: `models_for()` caches `SymbolModels` per `tf_entry`. The AF setup (`strategy/asia_fake.py`) is a separate detector appended by `SymbolModels.signals` when `cfg.asia_fake` is on.
- **Conservative execution** (`engine/backtest.py`): M1 simulation; a limit fills only when ask/bid actually touches it; if stop and target are both hit inside one bar the stop wins; the order is cancelled if TP is reached before the fill; stop-management modes `a|b|c|d` (v5/v6 use `c`); PRO daily limits; one position per pair.
- **Configuration is data**: the `Config` dataclass holds every knob. Layering order: `Config()` defaults -> `PRESETS[STRATEGY]` (a chain v1 -> v2 -> v3 -> v4 -> v5 -> v6; `v6` = v5 + `corr_second_risk_mult 0.5` is the default since round 12) -> named env vars (`RISK_PCT`, `DAILY_MAX_LOSS_PCT`, `DAILY_MAX_SL`, `MAX_LOSSES_SAME_PLACE`, `MIN_RR`, `FIXED_TP_R`, `STARTING_BALANCE`, `BE_MODE`, `REQUIRE_D1_BIAS`, `REQUIRE_PREMIUM_DISCOUNT`, `REQUIRE_POI_OR_MAJOR`; applied only when set) -> `.env` `CFG_<FIELD>=<json or text>` -> experiments file `base` -> variant `params`. `Config.replace()` rejects unknown keys. A new knob must default to OFF so existing presets and published results do not move.

### Results and metrics

- `backtest` writes `RESULTS_DIR/summary.txt`, `signals.csv`, `trades_<mode>.csv`.
- `experiments` writes `RESULTS_DIR/<run_id>/experiments.csv`, `report.md`, `config.json`, `<variant>/{signals.csv, trades_<mode>.csv, funnel.json}`, and `candidates.csv.gz` (every M5 Shift with all rejection reasons) only for variants with `"log": true` or `--log-all`. It also appends to `RESULTS_DIR/experiments_history.csv`.
- `OOS_FROM` (default 2026-01-01) splits in-sample / out-of-sample: choose on IS, confirm on OOS. `R_w`, `w_totalR`, `w_maxDD_R` are risk-weighted R (grade A x1.5, B x0.5, AF x0.5); the figures quoted in README and STRATEGY are the weighted ones.
- Signals skipped by a risk rule are logged in `trades_<mode>.csv` with status `blocked_daily` / `blocked_pair` / `blocked_corr` and a `shadow_R` (what the trade would have made). `metrics()` excludes them from `signals` and `fill%` and reports `blocked`, `blocked_shadow_R`. Anything else that reads a trades CSV must filter on `status == "filled"`.

### Adding a strategy idea (short form of ARCHITECTURE.md)

1. Add a `Config` field, default off.
2. Parameter-free feature -> stage A; rule or filter -> stage B with a `funnel` reason.
3. New signal columns propagate automatically to `signals.csv`, `trades_*.csv` and the report breakdowns.
4. Add a variant to `experiments.json`, run `experiments --only V5,<NAME>`.
5. Add a test in `tests/` using `smcbot.data.synthetic.make_m1` (no network, no real data).

## Conventions and gotchas

- `experiments.json` is the published round and stays untouched; each new research round gets its own `experiments_r<N>.json` (r11, r12, r13, r15, r17). The lab notebook is `RESEARCH_LOG.md` (Ukrainian): it records the acceptance rules fixed before each run, the runs' numbers and the verdicts. Older logs (`claude/backtest-log*.md`) live outside the repo (see below).
- `results_dk/` holds the Dukascopy runs. The shipped news calendar starts 2023, but `data/news/ff_calendar.csv` (from `smcbot news --since 2019-01-01`) covers 2019+ and takes priority. Round 15 headline: v6 has no edge on Dukascopy 2019–2023 (−8.8 R_w, 488 trades); the feed check on 2024–2026 gave +109.5 (Dukascopy) vs +157.7 (MT5) R_w, so quote the Dukascopy figure as the realistic one in future rounds. Round 16: no price-derived regime filter (ADR ratio, ADR, Kaufman ER, session expansion, rolling equity; thresholds p25/p50/p75) separates 2019–2023 from 2024–2026 (max t 1.2); do not re-test those. Round 17: liquidity strategies A–D (pool-to-pool, IFVG POI, D1/H4 gap POI, D1 sweep + M15 shift) failed the stage-0 gates or only filter existing v4 trades; D1-level pool targets are unreachable within the 12:00/16:55 NY day; do not re-test without a multi-day holding engine.
- Research discipline used since round 12: before coding a new filter/sizing rule, run an IS-only (signal_time < OOS_FROM) diagnostic on the latest V5 `trades_c.csv` and `candidates.csv.gz` (every M5 shift with `fwd_R2`) against a pre-registered gate; gates that fail mean the knob is not written. Acceptance for a variant: IS t >= 2 on R_w, OOS w_avgR and w_maxDD not worse than V5, better in >= 2 of 3 years, then a paired `neighbours` run with the variant's params as `base`.
- `af_be_at_r` (default 0): AF-only stop-to-entry after +X R; implemented in `simulate_trade` after the same bar's stop check (stop-first tie rule kept), exit kind `SL_BE`.
- Engine stress/risk knobs: `fill_through_pips` (limit fills only after price trades through the level; a 0.2/0.5 pip stress costs 3-8 % of v5's R) and `corr_second_risk_mult` (take the second same-direction pair at reduced risk instead of blocking; 0 = block as published).
- Comments cite course material (`L39`, `Q5`, `PRO`) and `claude/decisions.md`, `claude/methodology/*`, `claude/backtest-log*.md`. Those `claude/` files live in the user's Claude.ai project, not in this repo; do not search for them.
- `news/ff_calendar.csv` ships with the package; `DATA_DIR/news/ff_calendar.csv` (written by `smcbot news`) takes priority when present.
- `DATA_SOURCE=mt5` works only on Windows with a running terminal (`run_windows.bat`, `requirements-windows.txt`); `csv` is the Docker/Mac path; `dukascopy` downloads and caches by month under `DATA_DIR/cache`.
- `tradingview/smc_v6.pine` mirrors the strategy for charts and alerts but is deliberately simplified (no M3 sync, fixed spread, no daily limits, no cross-pair rule: the v6 x0.5 rule is only a reminder in the alert text). The bot is the source of truth for statistics. When strategy rules change, update the indicator and `STRATEGY.md` too.
- Presets are `course, v1–v6`; killzone sets are `user/extended/ny/silver_bullet/ext_asia`.
- The Docker `experiments` service mounts `experiments_r11/r12/r13/r15/r17.json` and `./results_dk`.
- `data/news/ff_calendar.csv` (2019+) matches the shipped calendar on all keys for 2023+, but 13 rows have revised `actual/forecast/previous`; only the `news_*` annotation columns of a re-run can differ (trading logic is unaffected while `news_mode` is off).
- Version lives in three places: `pyproject.toml`, `smcbot/__init__.py` (`__version__`), `CHANGELOG.md`.
