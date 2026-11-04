"""News calendar, entry models and filters, time stop, CFG_ overrides."""
import numpy as np
import pandas as pd
import pytest

from smcbot.config import Config, Instrument
from smcbot.market.news import annotate, load_calendar, news_filter, tier1_tag
from smcbot.strategy.setups import SideModel, SymbolModels
from smcbot.data.synthetic import make_m1

FF = """date,time,currency,impact,event,actual,forecast,previous
Fri Jan 5 2024,All Day,USD,,Bank Holiday,,,
Fri Jan 5 2024,13:30,USD,High,Non-Farm Employment Change,216K,168K,173K
Fri Jan 5 2024,15:00,USD,High,ISM Services PMI,50.6,52.6,52.7
Fri Jan 5 2024,10:00,EUR,Medium,CPI Flash Estimate y/y,2.9%,3.0%,2.4%
Fri Jan 5 2024,09:30,GBP,High,Construction PMI,46.8,46.0,45.5
Thu Jan 25 2024,13:15,EUR,High,Main Refinancing Rate,4.50%,4.50%,4.50%
"""


@pytest.fixture()
def cal(tmp_path):
    p = tmp_path / "ff.csv"
    p.write_text(FF, encoding="utf-8")
    return load_calendar(p)


def test_calendar_parse(cal):
    assert len(cal) == 6                       # All Day holiday kept as impact "Holiday"
    from smcbot.market.news import holidays
    assert holidays(cal)["USD"] == {"2024-01-05"}
    nfp = cal[cal["event"].str.startswith("Non-Farm")].iloc[0]
    assert nfp["time"] == pd.Timestamp("2024-01-05 13:30", tz="UTC")
    assert nfp["tag"] == "NFP" and tier1_tag("Main Refinancing Rate") == "ECB"


def test_annotate_and_filters(cal):
    ny = "America/New_York"
    sigs = pd.DataFrame({
        "symbol": ["EURUSD", "EURUSD", "GBPUSD", "EURUSD"],
        "signal_time": pd.to_datetime(["2024-01-05 08:10", "2024-01-05 08:45", "2024-01-05 04:40",
                                       "2024-01-10 03:00"]).tz_localize(ny),
    })
    sigs["ext_ns"] = sigs["signal_time"].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timedelta(minutes=10)
    a = annotate(sigs, cal)
    assert a.loc[0, "news_next_min"] == 20 and a.loc[0, "news_bucket"] == "pre_30"
    assert a.loc[1, "news_prev_min"] == 15 and a.loc[1, "news_bucket"] == "post_0_30"
    assert a.loc[2, "news_prev_min"] == 0 or a.loc[2, "news_prev"].startswith("GBP")   # GBP event 09:30 UTC
    assert a.loc[3, "news_bucket"] == "none" and a.loc[0, "news_day"] == "NFP+PMI"
    cfg = Config()
    from smcbot.market.news import holiday_filter
    sigs["killzone"] = ["NY", "NY", "LONDON", "LONDON"]
    kept, d = holiday_filter(sigs, cal, cfg.replace(holiday_mode="us_ny"))
    assert d == 2 and len(kept) == 2
    kept, d = holiday_filter(sigs, cal, cfg.replace(holiday_mode="any"))
    assert d == 3
    kept, d = news_filter(sigs, cal, cfg.replace(news_mode="block"))
    assert d == 3 and len(kept) == 1
    kept, _ = news_filter(sigs, cal, cfg.replace(news_mode="no_day", news_scope="tier1"))
    assert list(kept["signal_time"].dt.day) == [10]
    kept, _ = news_filter(sigs, cal, cfg.replace(news_mode="after", news_after_min=60))
    assert len(kept) == 2                      # 08:35 sweep after NFP, GBP sweep after 09:30 UTC PMI


def test_trade_news_in_trade(cal):
    tr = pd.DataFrame({"symbol": ["EURUSD"], "status": ["filled"],
                       "signal_time": [pd.Timestamp("2024-01-05 07:00", tz="America/New_York")],
                       "fill_time": [pd.Timestamp("2024-01-05 07:10", tz="America/New_York")],
                       "exit_time": [pd.Timestamp("2024-01-05 09:00", tz="America/New_York")]})
    a = annotate(tr, cal)
    assert "Non-Farm" in a.loc[0, "news_in_trade"]


@pytest.fixture(scope="module")
def sm():
    return SymbolModels(make_m1(days=90, seed=3), Config(), Instrument("EURUSD", default_spread_pips=1.0))


def test_new_knobs_are_subsets(sm):
    base = Config().replace(killzone_set="extended", require_pd=False, require_poi_or_major=False,
                                      require_d1_bias=False)
    s0 = sm.signals(base)
    for kw in ({"sweep_pools": ("ASIA",)}, {"po3_max_pips": 30}, {"killzone_set": "silver_bullet"}):
        s1 = sm.signals(base.replace(**kw))
        assert len(s1) <= len(s0)
        assert set(s1["signal_time"]).issubset(set(s0["signal_time"]))
    sc = sm.signals(base.replace(allow_continuation=True))
    assert len(sc) >= len(s0)
    assert set(s0["signal_time"]).issubset(set(sc["signal_time"]))
    assert len(s0) > 0 and len(sc) > len(s0)         # seed 3 / 90 days: 21 base, 24 with continuation
    assert sc["setup"].str.contains("S11").any()


def test_pick_pool_rules():
    sw = [("15min_SL", "MINOR", 1.0, 1), ("ASIA_L", "MEDIUM", 1.2, 2), ("LON_L", "MEDIUM", 1.1, 2)]
    c = Config()
    assert SideModel._pick_pool(sw, c)[0] == "ASIA_L"                       # LON hidden by default
    assert SideModel._pick_pool(sw, c.replace(session_pools=True))[0] == "LON_L"   # deeper of equal tier
    assert SideModel._pick_pool(sw, c.replace(sweep_pools=("PD",))) is None


def test_max_hold_time_stop():
    from smcbot.engine.backtest import simulate_trade
    idx = pd.date_range("2024-01-03 03:00", periods=600, freq="1min", tz="America/New_York")
    px = np.full(len(idx), 1.1000)
    m1 = pd.DataFrame({"open": px, "high": px + 0.0001, "low": px - 0.0001, "close": px, "spread": 0.0}, index=idx)
    sig = pd.Series({"side": 1, "entry": 1.1001, "sl": 1.0980, "tp": 1.1100, "signal_time": idx[0], "order": "market"})
    sw = pd.DataFrame({"kind": [], "known_at": [], "price": [], "time": []})
    inst = Instrument("EURUSD", default_spread_pips=0.0)
    r = simulate_trade(sig, m1, sw, Config().replace(max_hold_min=60), inst, "b")
    assert r["exit_kind"] == "HOLD" and abs(r["duration_min"] - 60) < 1.5
    r = simulate_trade(sig, m1, sw, Config(), inst, "b")
    assert r["exit_kind"] == "TIME"


def test_cfg_env_override(monkeypatch):
    from smcbot.settings import load_settings
    monkeypatch.setenv("CFG_NEWS_MODE", "block")
    monkeypatch.setenv("CFG_PO3_MAX_PIPS", "25")
    _, cfg = load_settings()
    assert cfg.news_mode == "block" and cfg.po3_max_pips == 25


def test_presets_and_strategy_env(monkeypatch):
    from smcbot.config import PRESETS
    from smcbot.settings import load_settings
    monkeypatch.setenv("STRATEGY", "v2")
    _, cfg = load_settings()
    assert cfg.allow_continuation and cfg.partial_at_r == 2.0 and cfg.killzone_set == "extended"
    assert set(PRESETS) >= {"course", "v1", "v2"}


def test_other_entry_timeframes_run():
    m1 = make_m1(days=40, seed=5)                    # loosened filters: 8 and 26 signals respectively
    for tfe in (("15min", "5min"), ("3min", "1min")):
        sm = SymbolModels(m1, Config().replace(tf_entry=tfe), Instrument("EURUSD", default_spread_pips=1.0))
        df = sm.signals(Config().replace(tf_entry=tfe, require_pd=False, require_poi_or_major=False,
                                         killzone_set="extended", require_d1_bias=False))
        assert sm.models[0].etf == tfe[0]
        assert len(df) > 0 and {"entry", "sl", "tp"} <= set(df.columns)


def test_signal_at_close_time_is_skipped():
    from smcbot.engine.backtest import simulate_trade
    idx = pd.date_range("2024-01-03 11:00", periods=300, freq="1min", tz="America/New_York")
    px = np.full(len(idx), 1.1)
    m1 = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "spread": 0.0}, index=idx)
    sig = pd.Series({"side": 1, "entry": 1.1, "sl": 1.098, "tp": 1.106,
                     "signal_time": pd.Timestamp("2024-01-03 12:00", tz="America/New_York"), "order": "market"})
    sw = pd.DataFrame({"kind": [], "known_at": [], "price": [], "time": []})
    r = simulate_trade(sig, m1, sw, Config(), Instrument("EURUSD", default_spread_pips=0.0), "b")
    assert r["status"] == "late"


def test_asia_signal_closes_next_day():
    from smcbot.engine.backtest import simulate_trade
    idx = pd.date_range("2024-01-03 20:00", periods=20 * 60, freq="1min", tz="America/New_York")
    px = np.full(len(idx), 1.1)
    m1 = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "spread": 0.0}, index=idx)
    sig = pd.Series({"side": 1, "entry": 1.1, "sl": 1.098, "tp": 1.106, "signal_time": idx[0], "order": "market"})
    sw = pd.DataFrame({"kind": [], "known_at": [], "price": [], "time": []})
    cfg = Config().replace(killzone_set="ext_asia")
    r = simulate_trade(sig, m1, sw, cfg, Instrument("EURUSD", default_spread_pips=0.0), "b")
    assert r["exit_kind"] == "TIME" and r["exit_time"] == pd.Timestamp("2024-01-04 12:00", tz="America/New_York")


def test_session_table_basic():
    from smcbot.research.sessions import session_table
    t = session_table(make_m1(days=20, seed=2))
    assert len(t) > 10
    for c in ("ASIA_dir", "FRA_x_ASIA", "LON_sweep_ASIA", "NY_x_LON", "day_high_by"):
        assert c in t
    assert set(t["FRA_x_ASIA"]) <= {"both", "high", "low", "none"}
    assert (t["DAY_range"] >= t["LON_range"] - 1e-9).all()


def test_slippage_makes_results_worse():
    from smcbot.engine.backtest import simulate_trade
    idx = pd.date_range("2024-01-03 03:00", periods=200, freq="1min", tz="America/New_York")
    px = np.r_[np.full(10, 1.1000), np.full(190, 1.0970)]
    m1 = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "spread": 0.0}, index=idx)
    sig = pd.Series({"side": 1, "entry": 1.1000, "sl": 1.0980, "tp": 1.1060, "signal_time": idx[0], "order": "market"})
    sw = pd.DataFrame({"kind": [], "known_at": [], "price": [], "time": []})
    inst = Instrument("EURUSD", default_spread_pips=0.0)
    r0 = simulate_trade(sig, m1, sw, Config(), inst, "b")["R_result"]
    r1 = simulate_trade(sig, m1, sw, Config().replace(slippage_pips=1.0), inst, "b")["R_result"]
    assert r1 < r0 <= -1 + 1e-9


def test_popular_entry_models_run(sm):
    base = Config().replace(killzone_set="extended", require_pd=False, require_poi_or_major=False,
                                      require_d1_bias=False)
    n0 = len(sm.signals(base))
    assert n0 > 0
    for kw in ({"entry_mode": "ote"}, {"entry_mode": "ifvg"}, {"require_unicorn": True},
               {"require_crt": "h4"}, {"fvg_min_retrace": 0.38, "fvg_max_retrace": 0.62}):
        df = sm.signals(base.replace(**kw))
        if "require_unicorn" not in kw:               # unicorn yields 0 signals on this series
            assert len(df) > 0
            assert {"entry", "sl", "tp", "unicorn", "crt_h4"} <= set(df.columns)
        if "require_crt" in kw or "require_unicorn" in kw or "fvg_min_retrace" in kw:
            assert len(df) <= n0


def test_grade_weighting(sm):
    base = Config().replace(killzone_set="extended", require_pd=False, require_poi_or_major=False,
                                      require_d1_bias=False)
    s0 = sm.signals(base)
    s1 = sm.signals(base.replace(grade_mode="weight"))
    s2 = sm.signals(base.replace(grade_mode="a_only"))
    assert len(s1) == len(s0) and len(s2) <= len(s0)
    assert len(s1) > 0 and len(s2) > 0               # seed 3 / 90 days: 11 A and 10 B signals
    assert set(s1["grade"]) == {"A", "B"}
    assert ((s1["grade"] == "A") == (s1["risk_mult"] == 1.5)).all()
    assert (s1["score"] >= 3).eq(s1["grade"] == "A").all()
    assert (s2["grade"] == "A").all()


def test_default_strategy_is_v6(monkeypatch):
    from smcbot.settings import load_settings
    monkeypatch.delenv("STRATEGY", raising=False)
    _, cfg = load_settings()
    assert cfg.grade_mode == "weight" and cfg.holiday_mode == "us_ny"
    assert cfg.asia_fake is True and cfg.cont_window_bars == 144
    assert cfg.corr_second_risk_mult == 0.5 and cfg.block_same_direction is True
