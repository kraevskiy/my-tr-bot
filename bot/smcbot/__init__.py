"""SMC bot for EURUSD / GBPUSD: ICT/SMC methodology formalized into a testable trading system.

Layers (each depends only on the ones above it):

    config / settings      what to trade and how (Config, presets, .env)
    data                   M1 bars from MT5 CSV / MT5 terminal / Dukascopy, time zones, resampling
    market                 sessions, structure, FVG/OB zones, liquidity pools, SMT, news calendar
    strategy               signals: sweep -> Shift (M5+M3) -> FVG limit, filters, setup grade A/B
    engine                 M1 execution simulation, stop management, PRO risk limits, metrics
    research               analytics reports, session study
    apps / cli             command-line entry points (python -m smcbot ...)
"""
__version__ = "3.4.0"
