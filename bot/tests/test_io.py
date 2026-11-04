import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcbot.data import providers  # noqa: E402
from smcbot.data.bars import load_mt5_csv  # noqa: E402
from smcbot.settings import load_settings  # noqa: E402


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RISK_PCT", "0.5")
    monkeypatch.setenv("SYMBOLS", "eurusd")
    monkeypatch.setenv("CSV_EURUSD", "/x.csv")
    monkeypatch.setenv("REQUIRE_D1_BIAS", "false")
    s, cfg = load_settings()
    assert cfg.risk_pct == 0.005 and s.symbols == ["EURUSD"]
    assert s.csv_paths["EURUSD"] == "/x.csv" and cfg.require_d1_bias is False


def test_mt5_csv_server_time_to_ny(tmp_path):
    p = tmp_path / "EURUSD_M1.csv"
    p.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                 "2024.03.05\t00:00:00\t1.08500\t1.08510\t1.08490\t1.08505\t50\t0\t7\n"
                 "2024.03.05\t00:01:00\t1.08505\t1.08520\t1.08500\t1.08515\t50\t0\t9\n")
    df = load_mt5_csv(str(p), "ny_close")
    assert df.index[0] == pd.Timestamp("2024-03-04 17:00", tz="America/New_York")   # = DO
    assert np.allclose(df["spread"], [0.7, 0.9])


def test_dukascopy_provider_with_cache(monkeypatch, tmp_path):
    calls = []

    def fake(code, a, b, side, retries=3):
        calls.append((code, side, a))
        idx = pd.date_range(a, b, freq="1min", inclusive="left", tz="UTC")
        base = 1.1 if side == "bid" else 1.10012
        return pd.DataFrame({"open": base, "high": base + 0.0002, "low": base - 0.0002,
                             "close": base, "volume": 1.0}, index=idx)

    monkeypatch.setattr(providers, "_fetch_dukascopy_month", fake)
    df = providers.load_dukascopy("EURUSD", "2024-01-01", "2024-03-01", str(tmp_path))
    assert str(df.index.tz) == "America/New_York"
    assert np.allclose(df["spread"], 1.2)
    assert not ((df.index.weekday == 5)).any()
    n_first = len(calls)
    providers.load_dukascopy("EURUSD", "2024-01-01", "2024-03-01", str(tmp_path))
    assert len(calls) == n_first          # second run served from the monthly cache


class FakeMT5:
    TIMEFRAME_M1 = 1

    def __init__(self):
        self.calls = 0

    def symbol_select(self, s, on):
        return s == "EURUSD.m"

    def symbol_info(self, s):
        return type("SI", (), {"point": 0.00001, "digits": 5})()

    def last_error(self):
        return (0, "ok")

    def copy_rates_range(self, s, tf, a, b):
        self.calls += 1
        t = pd.date_range(pd.Timestamp(a).tz_localize(None), pd.Timestamp(b).tz_localize(None),
                          freq="1min", inclusive="left")
        n = len(t)
        return pd.DataFrame({"time": (t - pd.Timestamp(0)) // pd.Timedelta("1s"), "open": 1.1, "high": 1.1002,
                             "low": 1.0998, "close": 1.1, "tick_volume": 1, "spread": 8,
                             "real_volume": 0}).to_records(index=False) if n else None


def test_mt5_provider(monkeypatch, tmp_path):
    from smcbot.data.mt5 import load_mt5
    monkeypatch.setenv("MT5_SYMBOL_EURUSD", "EURUSD.m")
    fake = FakeMT5()
    df = load_mt5("EURUSD", "2024-01-01", "2024-03-01", str(tmp_path), "ny_close", mt5=fake)
    # server 2024-01-02 00:00 (NY+7) -> NY 2024-01-01 17:00
    assert pd.Timestamp("2024-01-01 17:00", tz="America/New_York") in df.index
    assert np.allclose(df["spread"], 0.8)
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    calls = fake.calls
    load_mt5("EURUSD", "2024-01-01", "2024-03-01", str(tmp_path), "ny_close", mt5=fake)
    assert fake.calls == calls            # complete months come from cache


def test_csv_chunks_are_merged(tmp_path):
    from smcbot.data.providers import load_csv_files
    head = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
    (tmp_path / "EURUSD_M1_1.csv").write_text(head +
        "2024.03.05\t00:00:00\t1.1\t1.1\t1.1\t1.1\t1\t0\t7\n2024.03.05\t00:01:00\t1.1\t1.1\t1.1\t1.1\t1\t0\t7\n")
    (tmp_path / "EURUSD_M1_2.csv").write_text(head +
        "2024.03.05\t00:01:00\t1.1\t1.1\t1.1\t1.1\t1\t0\t7\n2024.03.05\t00:02:00\t1.1\t1.1\t1.1\t1.1\t1\t0\t7\n")
    df = load_csv_files(str(tmp_path / "EURUSD_M1*.csv"), "ny_close")
    assert len(df) == 3 and df.index.is_monotonic_increasing


def test_mt5_csv_utf16(tmp_path):
    p = tmp_path / "EURUSD_M1.csv"
    txt = ("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
           "2024.03.05\t00:00\t1.08500\t1.08510\t1.08490\t1.08505\t50\t0\t7\n")
    p.write_bytes(txt.encode("utf-16"))
    df = load_mt5_csv(str(p), "ny_close")
    assert len(df) == 1 and df["close"].iat[0] == 1.08505
