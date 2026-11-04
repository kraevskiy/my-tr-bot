"""Historical M1 bars straight from a running MetaTrader 5 terminal.

Requires the official `MetaTrader5` Python package, which exists for Windows only,
so this source runs natively on the Windows machine with MT5 (not inside Docker).

Env:
  MT5_PATH       optional path to terminal64.exe (otherwise the running/default terminal)
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
                 optional; if empty, the account already logged in the terminal is used
  MT5_SYMBOL_<SYM>  broker symbol name if it differs, e.g. MT5_SYMBOL_EURUSD=EURUSD.m
  MT5_SERVER_TZ  ny_close (server = NY+7h, typical) or an IANA zone name

Bar times from MT5 are the server's wall clock; they are converted to New York time.
Complete months are cached under DATA_DIR/cache_mt5/<SYMBOL>/YYYY-MM.csv.gz.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smcbot.data.bars import server_to_ny

log = logging.getLogger("smcbot.mt5")


def _mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "DATA_SOURCE=mt5 needs the MetaTrader5 package (Windows only): "
            "pip install MetaTrader5. On Linux/Mac/Docker use DATA_SOURCE=csv "
            "with files exported from MT5.") from e
    return mt5


def connect():
    mt5 = _mt5()
    kw = {}
    if os.getenv("MT5_PATH"):
        kw["path"] = os.environ["MT5_PATH"]
    if os.getenv("MT5_LOGIN"):
        kw["login"] = int(os.environ["MT5_LOGIN"])
        kw["password"] = os.getenv("MT5_PASSWORD", "")
        kw["server"] = os.getenv("MT5_SERVER", "")
    if not mt5.initialize(**kw):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    info = mt5.account_info()
    if info is not None:
        log.info("MT5 connected: account %s on %s", info.login, info.server)
    return mt5


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.normalize().replace(day=1)
    while cur < end:
        nxt = cur + pd.offsets.MonthBegin(1)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def load_mt5(symbol: str, start: str, end: str | None, data_dir: str,
             server_tz: str = "ny_close", mt5=None) -> pd.DataFrame:
    mt5 = mt5 or connect()
    broker_sym = os.getenv(f"MT5_SYMBOL_{symbol.upper()}", symbol.upper())
    if not mt5.symbol_select(broker_sym, True):
        raise RuntimeError(f"Symbol {broker_sym} not found in MT5 Market Watch "
                           f"(set MT5_SYMBOL_{symbol.upper()} to the broker's name)")
    si = mt5.symbol_info(broker_sym)
    point = si.point
    pip = point * 10 if si.digits in (3, 5) else point
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    cache = Path(data_dir) / "cache_mt5" / symbol.upper()
    cache.mkdir(parents=True, exist_ok=True)
    this_month = pd.Timestamp(datetime.now(timezone.utc)).normalize().replace(day=1)
    parts = []
    for a, b in _month_ranges(s, e):
        f = cache / f"{a:%Y-%m}.csv.gz"
        complete = a.replace(day=1) < this_month and b >= a.replace(day=1) + pd.offsets.MonthBegin(1)
        if f.exists() and complete:
            part = pd.read_csv(f, parse_dates=["server_time"])
        else:
            log.info("MT5: %s %s ...", broker_sym, f"{a:%Y-%m}")
            # widen by a day each side: request dates are UTC, bar times are server time
            rates = mt5.copy_rates_range(broker_sym, mt5.TIMEFRAME_M1,
                                         (a - pd.Timedelta(days=1)).to_pydatetime(),
                                         (b + pd.Timedelta(days=1)).to_pydatetime())
            if rates is None or len(rates) == 0:
                log.warning("MT5 returned no bars for %s %s (%s). Check Tools > Options > "
                            "Charts > 'Max bars in chart' = Unlimited.", broker_sym,
                            f"{a:%Y-%m}", mt5.last_error())
                continue
            r = pd.DataFrame(rates)
            part = pd.DataFrame({
                "server_time": pd.to_datetime(r["time"], unit="s"),
                "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
                "spread": r["spread"] * point / pip,
            })
            # keep only bars whose server time falls in this month (de-duplicates the widening)
            lo = a.tz_localize(None).replace(day=1)
            hi = lo + pd.offsets.MonthBegin(1)
            part = part[(part["server_time"] >= lo) & (part["server_time"] < hi)]
            if complete and len(part):
                part.to_csv(f, index=False, compression="gzip")
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "spread"])
    raw = pd.concat(parts, ignore_index=True).drop_duplicates("server_time").sort_values("server_time")
    idx = server_to_ny(raw["server_time"], server_tz)
    df = raw[["open", "high", "low", "close", "spread"]].set_index(pd.DatetimeIndex(idx, name="time"))
    df = df[~df.index.isna()]
    return df[~df.index.duplicated(keep="first")]
