import numpy as np
import pandas as pd

from smcbot.apps.prop import Rules, arrays, bootstrap, load, replay, rolling


def _csv(tmp_path, rows):
    """rows: (day_offset, R_w, mae_R=None). Exits at 10:00 NY (mixed EST/EDT offsets like the real files)."""
    out = []
    for d, r, *m in rows:
        ts = pd.Timestamp("2024-01-08 10:00", tz="America/New_York") + pd.Timedelta(days=d)
        out.append({"status": "filled", "exit_time": ts.isoformat(), "fill_time": (ts - pd.Timedelta(hours=1)).isoformat(),
                    "R_w": r, "mae_R": m[0] if m else 0.0, "risk_mult": 1.0, "R_result": r})
    out.append({"status": "skipped", "exit_time": None, "fill_time": None, "R_w": 99, "mae_R": 0, "risk_mult": 1, "R_result": 99})
    p = tmp_path / "t.csv"
    pd.DataFrame(out).to_csv(p, index=False)
    return p


def _run(tmp_path, rows, risk=1.0, **kw):
    return replay(*arrays(load(_csv(tmp_path, rows))), risk, Rules(**kw))


def test_two_phase_pass(tmp_path):
    p1 = [(0, 4), (1, 4), (2, 4), (3, 0.1)]
    r = _run(tmp_path, p1)
    assert r["result"] == "censored" and r["phase"] == 2 and r["d1"] == 3   # phase 1 passed on trading day 4
    # phase 2 starts from a fresh 100 (not 112): +1.25R x4 on days 4-7 -> 105 only on the 4th trading day
    rows = p1 + [(4, 1.25), (5, 1.25), (6, 1.25), (7, 1.25)]
    assert _run(tmp_path, rows[:-1])["result"] == "censored"
    r = _run(tmp_path, rows)
    assert r["result"] == "pass" and r["d_total"] == 7


def test_daily_fail(tmp_path):
    assert _run(tmp_path, [(0, -6)])["result"] == "fail_daily"
    assert _run(tmp_path, [(0, -3.1), (0, -3.1)])["result"] == "fail_daily"
    assert _run(tmp_path, [(0, -3.1)])["result"] == "censored"
    assert _run(tmp_path, [(0, -3.1), (1, -3.1)])["result"] == "censored"


def test_max_fail_not_daily(tmp_path):
    assert _run(tmp_path, [(0, -3.9), (1, -3.9), (2, -3.2)])["result"] == "fail_max"


def test_trailing_vs_static(tmp_path):
    # +9 -> 109; five -2 days -> 99 == peak_eod(109) - 10 -> trailing fails, static (floor 90) survives
    rows = [(0, 9)] + [(d, -2) for d in range(1, 6)]
    assert _run(tmp_path, rows)["result"] == "censored"
    assert _run(tmp_path, rows, trailing=True)["result"] == "fail_max"


def test_min_days(tmp_path):
    rows = [(0, 11), (1, 0.1), (2, 0.1), (3, 0.1)]
    assert _run(tmp_path, rows[:1])["phase"] == 1
    assert _run(tmp_path, rows[:3])["phase"] == 1
    r = _run(tmp_path, rows)
    assert r["phase"] == 2 and r["d1"] == 3
    r = _run(tmp_path, rows[:1], min_days=1)
    assert r["phase"] == 2 and r["d1"] == 0


def test_bootstrap_seed(tmp_path):
    rng = np.random.default_rng(0)
    rows = [(int(d), float(rng.choice([-1, -1, 2.5]))) for d in np.sort(rng.integers(0, 1080, 400))]
    arrs = arrays(load(_csv(tmp_path, rows)))
    a = bootstrap(arrs, [0.5, 1.0], Rules(), 300, 1, 24)
    pd.testing.assert_frame_equal(a, bootstrap(arrs, [0.5, 1.0], Rules(), 300, 1, 24))
    assert not a.equals(bootstrap(arrs, [0.5, 1.0], Rules(), 300, 2, 24))


def test_mae_check(tmp_path):
    rows = [(0, -3, 6.0)]
    assert _run(tmp_path, rows)["result"] == "censored"
    assert _run(tmp_path, rows, mae=True)["result"] == "fail_daily"


def test_rolling(tmp_path):
    rows = [(0, 4), (1, 4), (2, 4), (3, 0.1), (4, 1.25), (5, 1.25), (6, 1.25), (7, 1.25)]
    t = rolling(arrays(load(_csv(tmp_path, rows))), [1.0], Rules())
    n = 8   # 8 distinct starts; only the first can finish both phases
    assert t.loc[1.0, "P(pass)"] == 1 / n
    assert t.loc[1.0, "P(censored)"] == 7 / n   # last-day start is censored among them
