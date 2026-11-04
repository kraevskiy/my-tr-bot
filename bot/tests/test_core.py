import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcbot.engine.backtest import simulate_trade  # noqa: E402
from smcbot.config import NY_TZ, Config, Instrument  # noqa: E402
from smcbot.data.bars import TF_DELTA, resample  # noqa: E402
from smcbot.strategy.setups import generate_signals, mirror  # noqa: E402
from smcbot.market.structure import market_structure  # noqa: E402
from smcbot.data.synthetic import make_m1  # noqa: E402
from smcbot.market.zones import find_fvgs  # noqa: E402


def bars(rows, start="2024-03-05 02:00", freq="5min"):
    idx = pd.date_range(pd.Timestamp(start, tz=NY_TZ), periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_fvg_bull_and_bear():
    df = bars([(1.0, 1.1, 0.9, 1.05), (1.05, 1.5, 1.0, 1.45), (1.45, 1.6, 1.2, 1.55),
               (1.55, 1.56, 1.3, 1.32), (1.32, 1.33, 0.95, 1.0), (1.0, 1.05, 0.8, 0.85)])
    f = find_fvgs(df, TF_DELTA["5min"], 0.0)
    bull = f[f["dir"] == 1].iloc[0]
    assert (bull["lo"], bull["hi"]) == (1.1, 1.2)          # high[i-1] .. low[i+1]
    assert bull["known_at"] == df.index[2] + TF_DELTA["5min"]
    # closing below 1.1 (bar 4 closes 1.0) invalidates it at that bar's close
    assert bull["invalid_at"] == df.index[4] + TF_DELTA["5min"]
    bear = f[f["dir"] == -1]
    assert len(bear) and bear.iloc[0]["lo"] == 1.05 and bear.iloc[0]["hi"] == 1.3


def zigzag(points, per_leg=6):
    """Build bars that walk linearly through the given turning points."""
    px = []
    for a, b in zip(points[:-1], points[1:]):
        px += list(np.linspace(a, b, per_leg, endpoint=False))
    px.append(points[-1])
    rows = []
    for p0, p1 in zip(px[:-1], px[1:]):
        rows.append((p0, max(p0, p1) + 0.01, min(p0, p1) - 0.01, p1))
    return bars(rows)


def test_structure_bos_conf_sequence():
    # 2->1->3 high; pullback 2; close above 3 = BOS_UP (strong low 2);
    # pullback 3 after high 4; close above 4 = CONF_UP (strong low moves to 3);
    # high 5, pullback 3.5, close below 3 = BOS_DOWN
    df = zigzag([2, 1, 3, 2, 4, 3, 5, 3.5, 2.5])
    st, ev = market_structure(df, 2)
    kinds = [e.kind for e in ev]
    assert kinds[:3] == ["BOS_UP", "CONF_UP", "CONF_UP"] or kinds[:2] == ["BOS_UP", "CONF_UP"]
    i_conf = kinds.index("CONF_UP")
    assert ev[i_conf].first_conf
    assert "BOS_DOWN" in kinds[i_conf:]
    bos = [e for e in ev if e.kind == "BOS_DOWN"][0]
    assert bos.level == pytest.approx(3 - 0.01, abs=0.02)
    # BOS is triggered by a close, never by a wick alone
    for e in ev:
        if e.kind == "BOS_DOWN":
            assert df["close"].iat[e.pos] < e.level


def test_mirror_roundtrip():
    m = make_m1(days=3)
    mm = mirror(mirror(m))
    pd.testing.assert_frame_equal(m, mm)
    assert (mirror(m)["high"] >= mirror(m)["low"]).all()


def test_resample_day_boundary_is_17_ny():
    m = make_m1(days=10)
    d1 = resample(m, "1D")
    assert set(d1.index.hour) == {17}
    h4 = resample(m, "4h")
    assert set(h4.index.hour) <= {1, 5, 9, 13, 17, 21}


def test_no_lookahead_in_signals():
    """Signals before a cutoff must not change when later data is removed."""
    # loosened filters so the 60-day synthetic series (seed 3) yields signals
    cfg = Config().replace(killzone_set="extended", require_pd=False, require_poi_or_major=False,
                           require_d1_bias=False)
    inst = Instrument("EURUSD")
    full = make_m1(days=60, seed=3)
    cut = full.index[0] + pd.Timedelta(days=45)
    part = full[full.index < cut]
    a = generate_signals(full, cfg, inst)
    b = generate_signals(part, cfg, inst)
    lim = cut - pd.Timedelta(days=2)
    cols = ["signal_ns", "side", "entry", "sl", "tp", "pool", "setup"]
    a = a[a["signal_time"] < lim][cols].reset_index(drop=True)
    b = b[b["signal_time"] < lim][cols].reset_index(drop=True)
    assert len(a) > 0
    assert len(a) == len(b)
    pd.testing.assert_frame_equal(a, b)


def _m1_path(prices, start="2024-03-05 03:00"):
    idx = pd.date_range(pd.Timestamp(start, tz=NY_TZ), periods=len(prices), freq="1min")
    rows = []
    prev = prices[0]
    for p in prices:
        rows.append((prev, max(prev, p), min(prev, p), p, 0.0))
        prev = p
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "spread"], index=idx)


def _sig(t, side=1, entry=1.1000, sl=1.0990, tp=1.1040):
    return pd.Series({"side": side, "entry": entry, "sl": sl, "tp": tp,
                      "signal_time": pd.Timestamp(t, tz=NY_TZ)})


EMPTY_SW = pd.DataFrame({"kind": [], "known_at": np.array([], "datetime64[ns]"),
                         "price": [], "time": np.array([], "datetime64[ns]")})


@pytest.mark.parametrize("mode", list("abcd"))
def test_trade_tp_with_partial(mode):
    m1 = _m1_path([1.1005, 1.1000, 1.1005, 1.1010, 1.1020, 1.1030, 1.1041, 1.1041])
    r = simulate_trade(_sig("2024-03-05 03:00"), m1, EMPTY_SW, Config(), Instrument("EURUSD"), mode)
    assert r["status"] == "filled" and r["exit_kind"] == "TP"
    assert r["R_result"] == pytest.approx(0.5 * 1 + 0.5 * 4)


def test_trade_stop_before_partial():
    m1 = _m1_path([1.1005, 1.1000, 1.0995, 1.0989, 1.0980])
    r = simulate_trade(_sig("2024-03-05 03:00"), m1, EMPTY_SW, Config(), Instrument("EURUSD"), "b")
    assert r["exit_kind"] == "SL" and r["R_result"] == pytest.approx(-1)


def test_be_modes_after_partial():
    # fill, reach +1R (partial), come back to the original stop
    path = [1.1005, 1.1000, 1.1006, 1.1011, 1.1004, 1.0999, 1.0989]
    m1 = _m1_path(path)
    res = {m: simulate_trade(_sig("2024-03-05 03:00"), m1, EMPTY_SW, Config(), Instrument("EURUSD"), m)
           for m in "ab"}
    assert res["a"]["R_result"] == pytest.approx(0.5)      # stopped at entry
    assert res["b"]["R_result"] == pytest.approx(0.0)      # +0.5 - 0.5


def test_spread_blocks_long_fill():
    m1 = _m1_path([1.1005, 1.1001, 1.1003, 1.1002])
    m1["spread"] = 2.0          # ask = bid + 2 pips never reaches 1.1000
    r = simulate_trade(_sig("2024-03-05 03:00"), m1, EMPTY_SW, Config(), Instrument("EURUSD"), "b")
    assert r["status"] != "filled"


def test_mode_c_and_d_use_confirmed_swings_after_partial():
    # fill 03:01, +1R at 03:03, swing low 1.1004 at 03:04 confirmed 03:06, then dump
    path = [1.1005, 1.1000, 1.1006, 1.1011, 1.1004, 1.1008, 1.1009, 1.1003, 1.0999, 1.0985]
    m1 = _m1_path(path)
    t = m1.index
    sw = pd.DataFrame({"kind": ["low"], "price": [1.1004],
                       "time": [t[4].tz_convert("UTC").tz_localize(None).to_datetime64()],
                       "known_at": [t[6].tz_convert("UTC").tz_localize(None).to_datetime64()]})
    cfg, inst = Config(), Instrument("EURUSD")
    rc = simulate_trade(_sig("2024-03-05 03:00"), m1, sw, cfg, inst, "c")
    rd = simulate_trade(_sig("2024-03-05 03:00"), m1, sw, cfg, inst, "d")
    rb = simulate_trade(_sig("2024-03-05 03:00"), m1, sw, cfg, inst, "b")
    assert rb["R_result"] == pytest.approx(0.0)
    assert rc["R_result"] == pytest.approx(0.5)               # stop moved to entry
    # d: stop trailed to swing low - (0.5 pip buffer + 0 spread) = 1.10035 -> +0.175R on the rest
    assert rd["R_result"] == pytest.approx(0.5 + 0.5 * 0.35)


def test_config_replace_and_killzones():
    cfg = Config().replace(entry_mode="nearest", poi_tfs=["1h", "4h"], killzone_set="ny")
    assert cfg.entry_mode == "nearest" and cfg.poi_tfs == ("1h", "4h")
    assert cfg.killzones() == (("NY", "08:00", "10:00"),)
    with pytest.raises(KeyError):
        Config().replace(nope=1)


def test_market_order_and_mfe():
    m1 = _m1_path([1.1005, 1.1008, 1.1012, 1.1020, 1.1030, 1.1046, 1.1050])
    sig = _sig("2024-03-05 03:00", entry=1.1005, sl=1.0995, tp=1.1045)
    sig["order"] = "market"
    r = simulate_trade(sig, m1, EMPTY_SW, Config(), Instrument("EURUSD"), "b")
    assert r["status"] == "filled" and r["fill_price"] == pytest.approx(1.1005)
    assert r["exit_kind"] == "TP" and r["mfe_R"] >= 4 - 1e-9


def test_unfilled_signal_gets_shadow_result():
    m1 = _m1_path([1.1010, 1.1012, 1.1020, 1.1030, 1.1041, 1.1050])   # never back to 1.1000
    r = simulate_trade(_sig("2024-03-05 03:00"), m1, EMPTY_SW, Config(), Instrument("EURUSD"), "b")
    assert r["status"] == "missed" and r["shadow_mfe_R"] > 0


def test_variants_change_signals_without_recomputing():
    from smcbot.strategy.setups import SymbolModels
    cfg, inst = Config(), Instrument("EURUSD")
    sm = SymbolModels(make_m1(days=90, seed=4), cfg, inst)
    base = sm.signals(cfg)
    loose = sm.signals(cfg.replace(require_d1_bias=False, require_pd=False, require_poi_or_major=False))
    strict = sm.signals(cfg.replace(min_sl_pips=50))
    assert len(loose) >= len(base) >= len(strict) == 0
    logs = []
    sm.signals(cfg, log=logs)
    cl = logs[0]
    assert (cl["reasons"] == "").sum() == len(base)
