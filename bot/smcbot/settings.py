"""Runtime settings from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from smcbot.config import Config


def _f(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return float(v) if v else default


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    return default if not v else v in ("1", "true", "yes", "on")


@dataclass
class Settings:
    data_source: str = "csv"
    symbols: list = field(default_factory=lambda: ["EURUSD", "GBPUSD"])
    date_from: str = "2024-01-01"
    date_to: str | None = None
    data_dir: str = "/app/data"
    results_dir: str = "/app/results"
    csv_paths: dict = field(default_factory=dict)
    mt5_server_tz: str = "ny_close"
    dukascopy_fetch_ask: bool = True
    modes: str = "c"
    log_level: str = "INFO"


def load_settings() -> tuple[Settings, Config]:
    s = Settings(
        data_source=os.getenv("DATA_SOURCE", "csv").strip().lower(),
        symbols=[x.strip().upper() for x in os.getenv("SYMBOLS", "EURUSD,GBPUSD").split(",") if x.strip()],
        date_from=os.getenv("DATE_FROM", "2024-01-01").strip() or None,
        date_to=os.getenv("DATE_TO", "").strip() or None,
        data_dir=os.getenv("DATA_DIR", "/app/data"),
        results_dir=os.getenv("RESULTS_DIR", "/app/results"),
        mt5_server_tz=os.getenv("MT5_SERVER_TZ", "ny_close").strip(),
        dukascopy_fetch_ask=_b("DUKASCOPY_FETCH_ASK", True),
        modes=os.getenv("BE_MODES", "c").strip() or "c",
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
    s.csv_paths = {k[4:]: v for k, v in os.environ.items() if k.startswith("CSV_") and v}

    from smcbot.config import PRESETS
    preset = os.getenv("STRATEGY", "v6").strip().lower() or "v6"     # final strategy by default
    if preset not in PRESETS:
        raise ValueError(f"STRATEGY must be one of {sorted(PRESETS)}")
    cfg = Config().replace(**PRESETS[preset])
    # named env vars apply on top of the preset, only when set
    cfg.risk_pct = _f("RISK_PCT", cfg.risk_pct * 100) / 100
    cfg.daily_max_loss_pct = _f("DAILY_MAX_LOSS_PCT", cfg.daily_max_loss_pct * 100) / 100
    cfg.daily_max_sl = int(_f("DAILY_MAX_SL", cfg.daily_max_sl))
    cfg.max_losses_same_place = int(_f("MAX_LOSSES_SAME_PLACE", cfg.max_losses_same_place))
    cfg.min_rr = _f("MIN_RR", cfg.min_rr)
    cfg.fixed_tp_r_medium_minor = _f("FIXED_TP_R", cfg.fixed_tp_r_medium_minor)
    cfg.starting_balance = _f("STARTING_BALANCE", cfg.starting_balance)
    cfg.be_mode = os.getenv("BE_MODE", cfg.be_mode).strip().lower() or cfg.be_mode
    cfg.require_d1_bias = _b("REQUIRE_D1_BIAS", cfg.require_d1_bias)
    cfg.require_pd = _b("REQUIRE_PREMIUM_DISCOUNT", cfg.require_pd)
    cfg.require_poi_or_major = _b("REQUIRE_POI_OR_MAJOR", cfg.require_poi_or_major)
    # Generic overrides: CFG_<FIELD>=<json or text>, e.g. CFG_KILLZONE_SET=extended,
    # CFG_MIN_SL_PIPS=5, CFG_NEWS_MODE=block, CFG_SWEEP_POOLS=["ASIA","LON"]
    import json
    over = {}
    for k, v in os.environ.items():
        if k.startswith("CFG_") and v.strip():
            try:
                over[k[4:].lower()] = json.loads(v)
            except ValueError:
                over[k[4:].lower()] = v.strip()
    if over:
        cfg = cfg.replace(**over)
    return s, cfg
