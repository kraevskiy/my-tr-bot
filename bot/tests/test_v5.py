"""v5/v6: presets, BE mode default and the AF (Asia fake-out) setup."""
import numpy as np
import pandas as pd
import pytest

from smcbot.config import Config, Instrument, PRESETS
from smcbot.strategy.asia_fake import asia_fake_signals, build_m5
from smcbot.data.synthetic import make_m1


def test_v5_preset():
    c = Config().replace(**PRESETS["v5"])
    assert c.asia_fake and c.cont_window_bars == 144 and c.grade_mode == "weight"
    assert Config().be_mode == "c"


def test_v6_preset():
    import dataclasses
    v5 = dataclasses.asdict(Config().replace(**PRESETS["v5"]))
    v6 = dataclasses.asdict(Config().replace(**PRESETS["v6"]))
    assert v6.pop("corr_second_risk_mult") == 0.5
    assert v5.pop("corr_second_risk_mult") == 0.0
    assert v6 == v5


def _scenario(brk_at="02:30", gap_bars=0, drop_asia=None):
    """Flat Asia 1.1000-1.1010, wide pre-London range, fake breakout above at 02:30 and a strong reclaim.

    brk_at: NY time of the breakout M5 bar; gap_bars: M5 bars held above the Asia high between the
    breakout and the reclaim; drop_asia: (start, end) NY times of M1 bars removed from the Asia window.
    """
    idx = pd.date_range("2025-03-10 17:00", "2025-03-11 12:00", freq="1min", tz="America/New_York")
    base = np.full(len(idx), 1.1005)
    df = pd.DataFrame({"open": base, "high": base + 0.0002, "low": base - 0.0002, "close": base}, index=idx)
    t = df.index
    early = (t >= "2025-03-10 17:00") & (t < "2025-03-10 19:00")          # range before Asia: day range used
    df.loc[early, "high"] = 1.1060
    df.loc[early, "low"] = 1.0950
    asia = (t >= "2025-03-10 19:00") & (t < "2025-03-11 00:00")
    df.loc[asia, ["high"]] = 1.1010
    df.loc[asia, ["low"]] = 1.1000
    b0 = pd.Timestamp(f"2025-03-11 {brk_at}", tz="America/New_York")
    brk = (t >= b0) & (t < b0 + pd.Timedelta(minutes=5))                   # bullish M5 candle closing above Asia high
    df.loc[brk, "open"] = np.linspace(1.1004, 1.1025, brk.sum())
    df.loc[brk, "close"] = np.linspace(1.1006, 1.1030, brk.sum())
    df.loc[brk, "high"] = 1.1034
    df.loc[brk, "low"] = 1.1003
    if gap_bars:                                                           # stays above the Asia high: no reclaim yet
        gap = (t >= b0 + pd.Timedelta(minutes=5)) & (t < b0 + pd.Timedelta(minutes=5 * (1 + gap_bars)))
        df.loc[gap, ["open", "close"]] = 1.1030
        df.loc[gap, "high"] = 1.1031
        df.loc[gap, "low"] = 1.1029
    r0 = b0 + pd.Timedelta(minutes=5 * (1 + gap_bars))
    rec = (t >= r0) & (t < r0 + pd.Timedelta(minutes=5))                   # bearish reclaim back inside
    df.loc[rec, "open"] = np.linspace(1.1030, 1.1005, rec.sum())
    df.loc[rec, "close"] = np.linspace(1.1026, 1.0998, rec.sum())
    df.loc[rec, "high"] = 1.1031
    df.loc[rec, "low"] = 1.0997
    # history for ATR(D1): prepend 20 quiet days with 60-pip ranges
    hist = []
    for d in range(20, 0, -1):
        i2 = pd.date_range(idx[0] - pd.Timedelta(days=d), periods=24 * 60, freq="1min")
        x = pd.DataFrame({"open": 1.1005, "high": 1.1005 + 0.0030 * np.sin(np.arange(len(i2)) / 200),
                          "low": 1.1005 - 0.0030 * np.abs(np.sin(np.arange(len(i2)) / 150)), "close": 1.1005}, index=i2)
        x["high"] = x[["high", "open"]].max(axis=1)
        hist.append(x)
    if drop_asia:
        df = df.loc[~((t >= pd.Timestamp(f"2025-03-10 {drop_asia[0]}", tz="America/New_York"))
                      & (t < pd.Timestamp(f"2025-03-10 {drop_asia[1]}", tz="America/New_York")))]
    return pd.concat(hist + [df]).sort_index().loc[lambda z: ~z.index.duplicated()]


def test_af_short_signal():
    m1 = _scenario()
    cfg = Config().replace(**PRESETS["v5"])
    sig = asia_fake_signals(m1, cfg, Instrument("EURUSD", default_spread_pips=1.0))
    assert len(sig) == 1
    s = sig.iloc[0]
    assert s.side == -1 and s.setup == "AF" and s.order == "market"
    assert s.sl > 1.1034 and s.tp < s.entry
    assert abs((s.entry - s.tp) / (s.sl - s.entry) - 3.0) < 1e-6
    assert s.signal_time == pd.Timestamp("2025-03-11 02:40", tz="America/New_York")


INST = Instrument("EURUSD", default_spread_pips=1.0)


def _af(m1=None, **kw):
    m1 = _scenario() if m1 is None else m1
    return asia_fake_signals(m1, Config().replace(**PRESETS["v5"]).replace(**kw), INST)


def test_af_scenario_baseline_has_signal():
    assert len(_af()) == 1


def test_af_adr_min_blocks_signal():
    assert len(_af(af_adr_min=2.0)) == 0


def test_af_fade_bars_limits_reclaim_delay():
    m1 = _scenario(gap_bars=2)                      # reclaim is the 3rd bar after the breakout
    assert len(_af(m1, af_fade_bars=2)) == 0
    assert len(_af(m1, af_fade_bars=3)) == 1
    assert len(_af(m1, af_fade_bars=6)) == 1
    assert len(_af(af_fade_bars=1)) == 1            # baseline reclaims on the 1st bar


def test_af_min_sl_pips_floor():
    pip = INST.pip
    natural = _af().iloc[0]
    assert natural.risk_pips > 5.0                  # baseline stop is wider than the default floor
    s = _af(af_min_sl_pips=natural.risk_pips + 10).iloc[0]
    want = natural.risk_pips + 10
    assert abs(s.risk_pips - want) < 1e-6
    assert abs((s.sl - s.entry) / pip - want) < 1e-6
    assert abs(s.tp - (s.entry + s.side * 3.0 * s.risk)) < 1e-9
    assert abs(s.tp - (s.entry - 3.0 * want * pip)) < 1e-9


def test_af_body_k_blocks_small_breakout():
    assert len(_af(af_body_k=5.0)) == 0


def test_af_reclaim_body_k_blocks_weak_reclaim():
    assert len(_af(af_reclaim_body_k=5.0)) == 0


@pytest.mark.parametrize("brk_at,expected", [
    ("01:30", 0), ("01:55", 0), ("02:00", 1), ("04:50", 1), ("05:00", 0), ("05:30", 0)])
def test_af_breakout_window(brk_at, expected):
    assert len(_af(_scenario(brk_at=brk_at))) == expected


def test_af_asia_needs_80pct_of_bars():
    assert len(_af(_scenario(drop_asia=("20:00", "21:00")))) == 1      # 48 of 60 M5 bars: enough
    assert len(_af(_scenario(drop_asia=("20:00", "21:05")))) == 0      # 47 of 60: range rejected


def test_af_asia_end_changes_range():
    assert len(_af(af_asia_end="03:00")) == 0       # range now includes the breakout bar, nothing breaks out


def test_af_no_lookahead_on_scenario():
    m1 = _scenario()
    full = _af(m1)
    assert len(full) == 1
    cut = full.signal_time.iloc[0]
    for extra in (0, 30):                           # cut right at the signal and a bit later
        part = _af(m1.loc[:cut + pd.Timedelta(minutes=extra) - pd.Timedelta(minutes=1)])
        pd.testing.assert_frame_equal(part.reset_index(drop=True), full.reset_index(drop=True))
    assert len(_af(m1.loc[:cut - pd.Timedelta(minutes=6)])) == 0      # signal needs the reclaim bar


def test_af_no_lookahead_on_synthetic():
    m1 = make_m1(days=60, seed=3)                   # seed 3 / 60 days yields 5 AF signals
    cfg = Config().replace(**PRESETS["v5"])
    sig = asia_fake_signals(m1, cfg, Instrument("EURUSD"))
    assert len(sig) > 0
    t = pd.to_datetime(sig.signal_time).dt.tz_convert("America/New_York")
    mins = t.dt.hour * 60 + t.dt.minute
    assert ((mins > 120) & (mins <= 5 * 60 + 30)).all()
    assert (sig.groupby([t.dt.date]).size() <= 1).all()
    # truncating the data right after a signal must not change it
    cut = sig.signal_time.iloc[0]
    again = asia_fake_signals(m1.loc[:cut - pd.Timedelta(minutes=1)], cfg, Instrument("EURUSD"))
    assert len(again) and np.isclose(again.iloc[0].sl, sig.iloc[0].sl)


def test_af_be_at_r_in_signal_rows():
    cfg = Config().replace(**PRESETS["v6"])
    assert (asia_fake_signals(_scenario(), cfg, INST)["be_at_r"] == 0.0).all()
    sig = asia_fake_signals(_scenario(), cfg.replace(af_be_at_r=1.0), INST)
    assert len(sig) == 1 and (sig["be_at_r"] == 1.0).all()
