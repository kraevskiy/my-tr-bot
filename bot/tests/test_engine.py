"""Engine risk rules and order handling: run_backtest / simulate_trade / metrics on hand-built data."""
import numpy as np
import pandas as pd
import pytest

from smcbot.config import NY_TZ, Config, Instrument
from smcbot.engine.backtest import metrics, run_backtest, simulate_trade

DAY = "2025-03-10"
INSTS = {"EURUSD": Instrument("EURUSD", default_spread_pips=0.0),
         "GBPUSD": Instrument("GBPUSD", default_spread_pips=0.0)}


def _flat_m1(n=800):
    idx = pd.date_range(pd.Timestamp(f"{DAY} 00:00", tz=NY_TZ), periods=n, freq="1min")
    return pd.DataFrame({"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "spread": 0.0}, index=idx)


def _at(hhmm):
    return pd.Timestamp(f"{DAY} {hhmm}", tz=NY_TZ)


def _hit(m1, t, **vals):
    for col, v in vals.items():
        m1.loc[t, col] = v


def _sig(symbol="EURUSD", hhmm="02:00", side=1, order="market", pool_level=1.1, risk_mult=1.0,
         entry=None, sl=None, tp=None, partial_r=99.0):
    e = 1.1 if entry is None else entry
    risk = 0.001
    return dict(symbol=symbol, side=side, setup="T", signal_time=_at(hhmm), entry=e,
                sl=e - side * risk if sl is None else sl, tp=e + side * 3 * risk if tp is None else tp,
                risk=risk, rr=3.0, order=order, pool="P", pool_level=pool_level, tier="A",
                killzone="LONDON", grade=1.0, risk_mult=risk_mult, partial_r=partial_r)


def _no_swings():
    """Empty M5 swing table (flat data has none); only stop modes c/d read it."""
    return pd.DataFrame({"kind": pd.Series(dtype=object), "price": pd.Series(dtype=float),
                         "known_at": pd.Series(dtype="datetime64[ns]"), "time": pd.Series(dtype="datetime64[ns]")})


def _run(rows, m1_by_symbol, mode="b", **cfg_kw):
    cfg = Config().replace(**cfg_kw)
    swings = {k: _no_swings() for k in m1_by_symbol}
    return run_backtest(pd.DataFrame(rows), m1_by_symbol, cfg, INSTS, mode=mode, swings=swings)


def _m1s(m1):
    return {"EURUSD": m1.copy(), "GBPUSD": m1.copy()}


def _stop_at(m1s, hhmm, symbols=("EURUSD", "GBPUSD")):
    """Stop (1.0985 < sl 1.0990) hit 5 minutes after a signal at hhmm."""
    t = _at(hhmm) + pd.Timedelta(minutes=5)
    for s in symbols:
        _hit(m1s[s], t, low=1.0985)


def _filled_times(df, symbol="EURUSD"):
    f = df[(df["status"] == "filled") & (df["symbol"] == symbol)]
    return [t.strftime("%H:%M") for t in f["signal_time"]]


LOSS_TIMES = ["02:00", "02:30", "03:00", "03:30", "04:00"]


def _losing_day(times, **sig_kw):
    m1s = _m1s(_flat_m1())
    for t in times:
        _stop_at(m1s, t)
    return m1s, [_sig(hhmm=t, **(sig_kw(i) if callable(sig_kw) else sig_kw)) for i, t in enumerate(times)]


def test_daily_max_sl():
    m1s, rows = _losing_day(LOSS_TIMES[:3])
    df = _run(rows, m1s, daily_max_sl=2, daily_max_loss_pct=10.0, max_losses_same_place=99)
    assert _filled_times(df) == ["02:00", "02:30"]
    df = _run(rows, m1s, daily_max_sl=3, daily_max_loss_pct=10.0, max_losses_same_place=99)
    assert _filled_times(df) == ["02:00", "02:30", "03:00"]


def test_daily_max_loss_pct():
    m1s, rows = _losing_day(LOSS_TIMES[:3])
    df = _run(rows, m1s, risk_pct=0.003, daily_max_loss_pct=0.006, daily_max_sl=99,
              max_losses_same_place=99)
    assert _filled_times(df) == ["02:00", "02:30"]


def test_daily_max_loss_pct_weighted_by_risk_mult():
    m1s, rows = _losing_day(LOSS_TIMES, risk_mult=0.5)
    df = _run(rows, m1s, risk_pct=0.003, daily_max_loss_pct=0.006, daily_max_sl=99,
              max_losses_same_place=99)
    assert _filled_times(df) == LOSS_TIMES[:4]


def test_max_losses_same_place():
    m1s, rows = _losing_day(LOSS_TIMES[:3])        # identical pool_level 1.1
    df = _run(rows, m1s, max_losses_same_place=2, daily_max_loss_pct=10.0, daily_max_sl=99)
    assert _filled_times(df) == ["02:00", "02:30"]
    for i, r in enumerate(rows):
        r["pool_level"] = 1.1 + 0.01 * i
    df = _run(rows, m1s, max_losses_same_place=2, daily_max_loss_pct=10.0, daily_max_sl=99)
    assert _filled_times(df) == ["02:00", "02:30", "03:00"]


def test_one_per_pair():
    m1s = _m1s(_flat_m1())                         # flat: first trade stays open until 12:00
    df = _run([_sig(hhmm="02:00"), _sig(hhmm="03:00")], m1s)
    assert _filled_times(df) == ["02:00"]


def test_cross_pair_same_direction():
    m1s = _m1s(_flat_m1())
    df = _run([_sig("EURUSD", "02:00", side=1), _sig("GBPUSD", "03:00", side=1)], m1s)
    assert _filled_times(df, "GBPUSD") == []
    df = _run([_sig("EURUSD", "02:00", side=1), _sig("GBPUSD", "03:00", side=-1)], m1s)
    assert _filled_times(df, "GBPUSD") == ["03:00"]
    df = _run([_sig("EURUSD", "02:00", side=1), _sig("GBPUSD", "03:00", side=1)], m1s,
              block_same_direction=False)
    assert _filled_times(df, "GBPUSD") == ["03:00"]


def _limit_sig(hhmm="02:10"):
    return pd.Series(_sig(hhmm=hhmm, order="limit", entry=1.099, sl=1.098, tp=1.102))


def _sim(m1, cfg, sig, mode="b"):
    return simulate_trade(sig, m1, _no_swings(), cfg, INSTS["EURUSD"], mode)


def test_killzone_end_expiry():
    cfg = Config()                                  # London killzone ends 05:00 NY
    assert _sim(_flat_m1(), cfg, _limit_sig())["status"] == "expired"
    m1 = _flat_m1()
    _hit(m1, _at("04:30"), low=1.0985)              # touch inside the killzone: fills
    assert _sim(m1, cfg, _limit_sig())["status"] == "filled"
    m1 = _flat_m1()
    _hit(m1, _at("05:30"), low=1.0985)              # touch after killzone end: order is gone
    assert _sim(m1, cfg, _limit_sig())["status"] == "expired"


def test_order_expiry_min():
    cfg = Config().replace(order_expiry_min=30)
    m1 = _flat_m1()
    _hit(m1, _at("02:55"), low=1.0985)              # 45 min after the signal
    assert _sim(m1, cfg, _limit_sig())["status"] == "expired"
    assert _sim(m1, Config(), _limit_sig())["status"] == "filled"
    m1 = _flat_m1()
    _hit(m1, _at("02:30"), low=1.0985)              # 20 min after the signal
    assert _sim(m1, cfg, _limit_sig())["status"] == "filled"


def test_missed_vs_expired():
    cfg = Config()
    m1 = _flat_m1()
    _hit(m1, _at("02:30"), high=1.1025)             # TP 1.102 reached, entry 1.099 never touched
    assert _sim(m1, cfg, _limit_sig())["status"] == "missed"
    assert _sim(_flat_m1(), cfg, _limit_sig())["status"] == "expired"


def test_lock_r_after_partial():
    def trade(**kw):
        m1 = _flat_m1()
        _hit(m1, _at("02:05"), high=1.101)          # +1R: partial
        return _sim(m1, Config().replace(**kw), pd.Series(_sig(hhmm="02:00", partial_r=1.0)))
    base = trade()                                   # stop untouched: remainder rides to the 12:00 exit at entry
    assert base["R_result"] == pytest.approx(0.5)
    assert base["exit_kind"] == "TIME"
    locked = trade(lock_r_after_partial=0.5)         # remainder stopped at entry + 0.5R
    assert locked["exit_kind"] == "SL_AFTER_PARTIAL"
    assert locked["R_result"] == pytest.approx(0.5 + 0.5 * 0.5)


def test_slippage_time_exit():
    base = _sim(_flat_m1(), Config(), pd.Series(_sig()))
    assert base["R_result"] == pytest.approx(0.0)
    slipped = _sim(_flat_m1(), Config().replace(slippage_pips=1.0), pd.Series(_sig()))
    # entry +1 pip (risk grows to 11 pips), time exit -1 pip: 2 pips lost
    assert slipped["exit_kind"] == "TIME"
    assert slipped["R_result"] == pytest.approx(-2 / 11)


def test_metrics_oos_split():
    st = lambda d: pd.Timestamp(d, tz=NY_TZ)
    tr = pd.DataFrame({
        "status": ["filled", "filled", "filled", "filled", "missed"],
        "R_result": [1.0, -1.0, 2.0, -1.0, np.nan],
        "risk_mult": [1.0, 0.5, 1.0, 0.5, 1.0],
        "signal_time": [st("2025-12-31 03:00"), st("2025-12-31 04:00"), st("2026-01-01 03:00"),
                        st("2026-01-02 03:00"), st("2026-01-02 04:00")],
    })
    tr["R_w"] = tr["R_result"] * tr["risk_mult"]
    m = metrics(tr, "2026-01-01")
    assert m["signals"] == 5 and m["fill%"] == pytest.approx(80.0)
    assert m["IS_trades"] == 2 and m["OOS_trades"] == 2
    assert m["IS_totalR"] == pytest.approx(0.0) and m["OOS_totalR"] == pytest.approx(1.0)
    assert m["w_totalR"] == pytest.approx(2.0)          # 1 - 0.5 + 2 - 0.5
    assert m["w_maxDD_R"] == pytest.approx(0.5)         # equity 1, 0.5, 2.5, 2.0
    assert m["t_stat"] == pytest.approx(0.25 / (1.5 / 2))


def test_mode_c_runs_through_engine():
    m1s = _m1s(_flat_m1())
    df = _run([_sig()], m1s, mode="c")
    assert df.loc[0, "status"] == "filled" and df.loc[0, "mode"] == "c"


def test_blocked_rows_logged_and_excluded():
    m1s = _m1s(_flat_m1())
    _hit(m1s["EURUSD"], _at("03:05"), high=1.1035)   # the blocked 03:00 EURUSD signal would hit its target
    _hit(m1s["GBPUSD"], _at("03:05"), high=1.1035)
    # EURUSD is open (flat to 12:00): same-pair repeat -> blocked_pair, GBPUSD long -> blocked_corr
    rows = [_sig("EURUSD", "02:00"), _sig("EURUSD", "03:00"), _sig("GBPUSD", "03:00")]
    df = _run(rows, m1s)
    assert sorted(df["status"]) == ["blocked_corr", "blocked_pair", "filled"]
    assert df.loc[df["status"] != "filled", "shadow_R"].round(6).tolist() == [3.0, 3.0]
    m = metrics(df)
    assert m["signals"] == 1 and m["fill%"] == pytest.approx(100.0)
    assert m["blocked"] == 2 and m["blocked_shadow_R"] == pytest.approx(6.0)
    # day stopped: the third loss of the day is logged as blocked_daily, balance untouched
    m1s, rows = _losing_day(LOSS_TIMES[:3])
    df = _run(rows, m1s, daily_max_sl=2, daily_max_loss_pct=10.0, max_losses_same_place=99)
    assert df["status"].tolist() == ["filled", "filled", "blocked_daily"]
    assert df.loc[2, "shadow_R"] == pytest.approx(-1.0)
    assert df["balance"].dropna().iloc[-1] == df.loc[1, "balance"]


def test_corr_second_risk_mult():
    m1s = _m1s(_flat_m1())                         # flat: both trades stay open until the close
    rows = [_sig("EURUSD", "02:00", side=1), _sig("GBPUSD", "03:00", side=1, risk_mult=2.0)]
    off = _run(rows, m1s)
    assert off.loc[off["symbol"] == "GBPUSD", "status"].iloc[0] == "blocked_corr"
    ref = _run(rows, m1s, block_same_direction=False)
    on = _run(rows, m1s, corr_second_risk_mult=0.5)
    g, g_ref = (d[d["symbol"] == "GBPUSD"].iloc[0] for d in (on, ref))
    assert g["status"] == "filled" and g["corr_scaled"]
    assert g["risk_mult"] == pytest.approx(0.5 * 2.0)
    assert g["R_w"] == pytest.approx(0.5 * 2.0 * g["R_result"])
    assert g["lots"] == pytest.approx(g_ref["lots"] / 2, abs=0.011)
    e = on[on["symbol"] == "EURUSD"].iloc[0]
    assert not e["corr_scaled"] and e["risk_mult"] == 1.0
    # opposite direction and block_same_direction=False ignore the knob
    assert not _run(rows, m1s, block_same_direction=False, corr_second_risk_mult=0.5)["corr_scaled"].any()

    # daily-loss tracker sees the scaled loss: full + half = 1.5R * 0.3% < 0.5%; two full = 0.6% >= 0.5%
    m1s = _m1s(_flat_m1())
    _hit(m1s["EURUSD"], _at("02:20"), low=1.0985)
    _hit(m1s["GBPUSD"], _at("02:15"), low=1.0985)
    rows = [_sig("EURUSD", "02:00"), _sig("GBPUSD", "02:10"), _sig("EURUSD", "03:00")]
    kw = dict(risk_pct=0.003, daily_max_loss_pct=0.005)
    df = _run(rows, m1s, corr_second_risk_mult=0.5, **kw)
    assert df["status"].tolist() == ["filled", "filled", "filled"]
    df = _run(rows, m1s, block_same_direction=False, **kw)
    assert df["status"].tolist() == ["filled", "filled", "blocked_daily"]


def test_fill_through():
    sig = _limit_sig()                              # long limit at 1.099, spread 0
    thru = Config().replace(fill_through_pips=0.5)
    m1 = _flat_m1()
    _hit(m1, _at("02:30"), low=1.099)               # exact touch
    assert _sim(m1, Config(), sig)["status"] == "filled"
    assert _sim(m1, thru, sig)["status"] == "expired"
    m1 = _flat_m1()
    _hit(m1, _at("02:30"), low=1.09894)             # trades through by 0.6 pip
    assert _sim(m1, Config(), sig)["status"] == "filled"
    r = _sim(m1, thru, sig)
    assert r["status"] == "filled" and r["fill_price"] == pytest.approx(1.099)   # fill price stays at entry
    m1 = _flat_m1()
    _hit(m1, _at("02:30"), low=1.099)
    _hit(m1, _at("02:40"), high=1.1025)             # touch only, then TP
    assert _sim(m1, Config(), sig)["status"] == "filled"
    assert _sim(m1, thru, sig)["status"] == "missed"


def test_new_engine_knobs_default_off():
    from smcbot.config import PRESETS
    for cfg in (Config(), Config().replace(**PRESETS["v5"])):
        assert cfg.corr_second_risk_mult == 0.0 and cfg.fill_through_pips == 0.0
