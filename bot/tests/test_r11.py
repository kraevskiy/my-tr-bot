"""Round 11 knobs: kz_risk_mult, af_symbols, block_same_direction (defaults keep old behaviour)."""
import pytest

from smcbot.config import PRESETS, Config, Instrument
from smcbot.data.synthetic import make_m1
from smcbot.strategy.setups import SymbolModels


def test_kz_risk_mult_scales_only_that_killzone():
    cfg = Config().replace(killzone_set="extended", require_pd=False, require_poi_or_major=False,
                           require_d1_bias=False, grade_mode="off")
    sm = SymbolModels(make_m1(days=90, seed=3), cfg, Instrument("EURUSD", default_spread_pips=1.0))
    base = sm.signals(cfg)
    half = sm.signals(cfg.replace(kz_risk_mult={"NY": 0.5}))
    assert len(base) == len(half) > 0
    assert set(base["killzone"]) >= {"NY"} and len(set(base["killzone"])) > 1
    key = ["signal_time", "side"]
    b = base.sort_values(key).reset_index(drop=True)
    h = half.sort_values(key).reset_index(drop=True)
    ny = b["killzone"] == "NY"
    assert (h.loc[ny, "risk_mult"] == b.loc[ny, "risk_mult"] * 0.5).all()
    assert (h.loc[~ny, "risk_mult"] == b.loc[~ny, "risk_mult"]).all()


def test_af_symbols_filters_af():
    cfg = Config().replace(**PRESETS["v5"])
    sm = SymbolModels(make_m1(days=60, seed=3), cfg, Instrument("EURUSD"))

    def af(c):
        s = sm.signals(c)
        return s[s["setup"] == "AF"]

    allaf = af(cfg)
    assert len(allaf) > 0
    assert len(af(cfg.replace(af_symbols=("GBPUSD",)))) == 0
    assert af(cfg.replace(af_symbols=["EURUSD"])).equals(allaf)


def test_new_knobs_default_off():
    for cfg in (Config(), Config().replace(**PRESETS["v5"])):
        assert cfg.kz_risk_mult == {}
        assert cfg.af_symbols == ()
        assert cfg.block_same_direction is True
    assert Config().replace(kz_risk_mult={"ASIA": 0.5}).kz_risk_mult == {"ASIA": 0.5}
