"""Strategy configuration.

Every number here comes from the user's decisions (claude/decisions.md) or from the
course methodology (claude/methodology/*). Values marked PROPOSAL are defaults that the
course does not fix; they are parameters to tune in backtests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

NY_TZ = "America/New_York"
KYIV_TZ = "Europe/Kyiv"


@dataclass
class Instrument:
    symbol: str
    pip: float = 0.0001
    pip_value_per_lot: float = 10.0      # USD per pip per 1.0 lot for xxxUSD pairs (L7, L15)
    lot_step: float = 0.01
    default_spread_pips: float = 1.0     # used when the data has no spread column


@dataclass
class Config:
    # --- timeframes (Q2) ---
    tf_trend: str = "4h"          # H4 trend (BOS/Conf state machine)
    tf_bias: str = "1D"           # D1 bias; W1 used as an extra factor
    tf_pd: str = "1h"             # H1 Premium/Discount
    tf_poi: tuple = ("15min", "1h", "4h")   # POI timeframes (M15 primary)
    tf_entry: tuple = ("5min", "3min")      # Shift must appear on both (sync)

    # --- swings (PROPOSAL: n not fixed by the course) ---
    fractal_n: dict = field(default_factory=lambda: {
        "3min": 2, "5min": 2, "15min": 2, "1h": 3, "4h": 3, "1D": 3, "1W": 2})

    # --- killzones in New York time (Q3: London 09-12, NY 15-17 Kyiv) ---
    killzones_ny: tuple = (("LONDON", "02:00", "05:00"), ("NY", "08:00", "10:00"))
    asia_ny: tuple = ("19:00", "02:00")     # 02:00-09:00 Kyiv (L26 webinar), PROPOSAL Q18
    cbdr_ny: tuple = ("16:00", "20:00")     # 23:00-03:00 Kyiv (L30)

    # --- filters ---
    require_d1_bias: bool = True            # D1 trend must agree with the trade
    require_pd: bool = True                 # long only in H1 discount, short only in premium
    require_poi_or_major: bool = True       # sweep must tag an HTF POI or a Major pool
    sweep_return_bars: int = 3              # PROPOSAL Q9: close back within <=3 bars
    sweep_lookback_bars: int = 36           # M5 bars between sweep and Shift (3h)
    sync_window_bars: int = 6               # M5 bars: M3 Shift must be within this window
    eq_tol_pips: float = 2.0                # PROPOSAL EQH/EQL tolerance
    min_fvg_pips: float = 0.5               # PROPOSAL min FVG size on M1-M5
    ob_max_engulf_bars: int = 3             # PROPOSAL
    manip_max_pips: float = 30.0            # L42: manipulation beyond NYM/DO <= 30 pips (factor)

    # --- exits (Q5, Q6, PRO) ---
    min_rr: float = 2.0                     # Q6
    fixed_tp_r_medium_minor: float = 4.0    # PRO: Medium/Minor liquidity -> fixed +4R
    partial_at_r: float = 1.0               # PRO: 50% at 1R
    partial_frac: float = 0.5
    be_mode: str = "c"                      # a|b|c|d (Q5), backtest compares all
    sl_buffer_pips: float = 0.5             # PROPOSAL: buffer = spread + 0.5 pip
    close_time_ny: str = "12:00"            # PROPOSAL Q22: close leftovers 19:00 Kyiv
    order_expiry: str = "killzone_end"

    # --- strategy variant knobs (tested with python -m smcbot experiments) ---
    entry_mode: str = "fvg_edge"         # fvg_edge | fvg_ce | ob_open | nearest | market | retest | retest_fvg | ote | ifvg
    order_expiry_min: int = 0            # 0 = until killzone end, else minutes after signal
    min_sl_pips: float = 0.0             # 0 = off
    max_sl_pips: float = 0.0             # 0 = off
    require_m3_sync: bool = True
    poi_tfs: tuple = ("15min", "1h", "4h")
    allow_s1: bool = True                # with H4 trend
    allow_s2: bool = True                # counter-trend after M15 BOS
    killzone_set: str = "user"           # user (09-12, 15-17 Kyiv) | extended (course: 09-14, 14-19 Kyiv)
    # pool-based targets: majorcap = take profit just before a MAJOR pool closer than tp_base_r;
    # majorext = stretch the target to a MAJOR pool between tp_base_r and pool_rr_max
    tp_base_r: float = 3.0
    target_pools: tuple = ()             # name prefixes for target pools, e.g. ("PW","PM"); () = all MAJOR
    pool_rr_min: float = 1.5
    pool_rr_max: float = 6.0
    pool_front_frac: float = 0.95        # exit at 95% of the distance (in front of the level)
    partial_mode: str = "r"              # r = at partial_at_r | pool = at a MAJOR pool if it is closer
    tp_mode: str = "pro"                 # pro | pool | fixedX (e.g. fixed2, fixed2.5, fixed3)
    bias_mode: str = "h4"                # h4: H4 trend (+D1) / M15 BOS for S2 | daily_candle: L39 candle model | none
    smt_mode: str = "off"                # off (annotate only) | require: EURUSD/GBPUSD divergence at the sweep
    max_spread_frac: float = 0.0         # skip if spread / risk > this (0 = off), e.g. 0.12
    require_asia_london: bool = False    # in the London killzone only take sweeps of the Asia range (S3 strict)
    partial_enabled: bool = True
    # --- v0.7 new entry models / filters ---
    session_pools: bool = False          # add London session high/low (LON_H/LON_L, 02-05 NY) as MEDIUM pools
    sweep_pools: tuple = ()              # only these swept-pool name prefixes, e.g. ("ASIA","LON") = Session Raid; () = all
    po3_max_pips: float = 0.0            # PO3 (L42): sweep must be below the NY-midnight open by <= X pips (0 = off)
    grade_mode: str = "off"              # off | weight (risk x grade_risk_a / _b) | a_only
    grade_po3_pips: float = 25.0
    grade_delay_min: float = 40.0
    grade_risk_a: float = 1.5
    grade_risk_b: float = 0.5
    kz_risk_mult: dict = field(default_factory=dict)   # killzone label -> risk multiplier (e.g. {"ASIA": 0.5}); {} = off
    po3_soft: bool = False               # po3_max_pips only limits the distance sweep <-> NY-midnight open
    avoid_candle_against: bool = False   # skip if the L39 daily candle model points against the trade
    require_m15_align: bool = False      # M15 structure must already point in the trade direction
    allow_continuation: bool = False     # S11: Shift without a new sweep, higher low after a recent swept leg
    cont_window_bars: int = 72           # M5 bars (<=288) back to look for the swept leg (S11)
    ote_level: float = 0.705             # entry_mode=ote: retracement of the Shift leg (0.62 / 0.705 / 0.79)
    fvg_min_retrace: float = 0.0         # FVG edge must sit at >= X retracement of the leg (OTE confluence), 0 = off
    fvg_max_retrace: float = 0.0         # FVG edge must sit at <= X retracement of the leg, 0 = off
    require_unicorn: bool = False        # FVG must overlap the broken swing level (breaker + FVG = Unicorn)
    require_crt: str = "off"             # off | h4 | h1 | any: last closed candle swept the previous one and closed inside
    max_adr_used: float = 0.0            # skip if today's range before the signal > X * ADR(10) (0 = off)
    max_delay_min: int = 0               # skip if sweep -> signal took longer than this (0 = off)
    slippage_pips: float = 0.0           # stress test: extra pips lost on stops, market entries and time exits
    exit_on_fvg_fail: bool = False       # exit early if an M5 candle closes beyond the far side of the entry FVG
    fvg_fail_buffer_pips: float = 0.0
    lock_r_after_partial: float = 0.0    # after the partial: stop to entry + X R (0 = per stop mode)
    max_hold_min: int = 0                # time stop after fill, minutes (0 = off)
    holiday_mode: str = "off"            # off | us_ny (no NY-killzone trades on US bank holidays) | any (no trades
                                         #   when either currency of the pair has a bank holiday) | any_ny
    news_mode: str = "off"               # off | block | after | no_day  (smcbot/market/news.py)
    news_scope: str = "high"             # high = all HIGH impact of the pair's currencies | tier1 = NFP/CPI/FOMC/ECB/BOE...
    news_before_min: int = 30            # block: no signal this many minutes BEFORE an event
    news_after_min: int = 30             # block: ...or this many minutes AFTER; after-mode: sweep within X min after
    shadow_hours: float = 4.0            # horizon for MFE/MAE of non-filled signals

    # --- setup AF "Asia fake-out at the London open" (v5; smcbot/strategy/asia_fake.py) ---
    asia_fake: bool = False              # add AF signals to the v4 signals
    asia_fake_only: bool = False         # research: only AF signals (no v4)
    af_asia_start: str = "19:00"         # Asia range, NY time (wraps midnight)
    af_asia_end: str = "00:00"
    af_win_start: str = "02:00"          # breakout window, NY time (the edge exists only from 02:00)
    af_win_end: str = "05:00"
    af_body_k: float = 0.5               # breakout candle body >= k * ATR14(M5)
    af_fade_bars: int = 6                # close back inside within N M5 bars
    af_reclaim_body_k: float = 0.5       # reclaim candle body >= k * ATR14(M5)
    af_adr_min: float = 0.40             # day range before the breakout >= x * ATR14(D1)
    af_tp_r: float = 3.0
    af_sl_buffer_pips: float = 1.0
    af_min_sl_pips: float = 5.0
    af_symbols: tuple = ()               # AF only on these symbols, e.g. ("EURUSD",); () = all
    af_risk_mult: float = 0.5            # risk of an AF trade relative to the base risk

    # --- risk (Q4: PRO rules) ---
    risk_pct: float = 0.003
    daily_max_loss_pct: float = 0.015
    daily_max_sl: int = 5
    max_losses_same_place: int = 3
    max_open_per_pair: int = 1
    block_same_direction: bool = True    # PRO rule: no second pair open in the same direction; False = off
    corr_second_risk_mult: float = 0.0   # 0 = off (same-direction cross-pair signals stay blocked); >0 = take them at risk_mult * X
    fill_through_pips: float = 0.0       # stress test: a limit fills only after price trades THROUGH the level by X pips (fill price stays entry)
    starting_balance: float = 10_000.0

    def killzones(self) -> tuple:
        sets = {
            "user": self.killzones_ny,
            "extended": (("LONDON", "02:00", "07:00"), ("NY", "07:00", "12:00")),
            "london": (self.killzones_ny[0],),
            "ny": (self.killzones_ny[1],),
            "london_ext": (("LONDON", "02:00", "07:00"),),
            "ny_ext": (("NY", "07:00", "12:00"),),
            # ICT Silver Bullet windows (London 03-04, NY AM 10-11 New York time)
            "silver_bullet": (("LONDON", "03:00", "04:00"), ("NY", "10:00", "11:00")),
            "sb_ny": (("NY", "10:00", "11:00"),),
            "asia": (("ASIA", "19:00", "23:59"),),
            "ext_asia": (("ASIA", "19:00", "23:59"), ("LONDON", "02:00", "07:00"), ("NY", "07:00", "12:00")),
            "ny_pm": (("NYPM", "13:00", "15:30"),),
            "asia_early": (("ASIA", "18:00", "21:00"),),
            "asia_late": (("ASIA", "21:00", "23:59"),),
        }
        if self.killzone_set not in sets:
            raise ValueError(f"killzone_set must be one of {list(sets)}")
        return sets[self.killzone_set]

    def replace(self, **kw) -> "Config":
        import dataclasses
        bad = set(kw) - {f.name for f in dataclasses.fields(self)}
        if bad:
            raise KeyError(f"Unknown config keys: {sorted(bad)}")
        return dataclasses.replace(self, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})


# Frozen strategy versions (see claude/backtest-log.md). Select with STRATEGY=v2 in .env
# or "preset": "v2" in experiments.json; CFG_* / variant params still override on top.
PRESETS = {
    "course": {},
    "v1": {"killzone_set": "extended", "min_sl_pips": 5, "tp_mode": "fixed3",
           "require_d1_bias": False, "max_spread_frac": 0.12},
}
PRESETS["v2"] = {**PRESETS["v1"], "allow_continuation": True, "partial_at_r": 2.0}
PRESETS["v3"] = {**PRESETS["v2"], "holiday_mode": "us_ny"}      # no NY-killzone trades on US bank holidays
PRESETS["v4"] = {**PRESETS["v3"], "grade_mode": "weight"}        # candidate: A/B setup grading, risk 1.5x / 0.5x
PRESETS["v5"] = {**PRESETS["v4"], "cont_window_bars": 144, "asia_fake": True}   # final: v4 + AF (BE mode c)
PRESETS["v6"] = {**PRESETS["v5"], "corr_second_risk_mult": 0.5}   # v5 + second correlated pair at half risk (round 12)
PRESETS["v5_core"] = {**PRESETS["v4"], "cont_window_bars": 144}                  # v5 without AF
PRESETS["af_only"] = {**PRESETS["v4"], "asia_fake": True, "asia_fake_only": True}  # AF alone (research)
PRESETS["v3_asia"] = {**PRESETS["v3"], "killzone_set": "ext_asia"}  # candidate v4: + Asia window 19:00-24:00 NY
