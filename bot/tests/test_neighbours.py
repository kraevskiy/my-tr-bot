import json

import pandas as pd

from smcbot.apps import neighbours as nbm
from smcbot.config import PRESETS, Config

BASE = Config().replace(**PRESETS["v5"])


def test_deterministic_and_seed_sensitive():
    a, b, c = nbm.generate(10, 0.2, 7, "v5"), nbm.generate(10, 0.2, 7, "v5"), nbm.generate(10, 0.2, 8, "v5")
    assert json.dumps(a) == json.dumps(b)
    assert a["variants"][1]["params"] != c["variants"][1]["params"]


def test_params_within_bounds_and_valid():
    spec = json.loads(json.dumps(nbm.generate(50, 0.2, 3, "v5")))     # must serialise
    for v in spec["variants"][1:]:
        p = v["params"]
        for k in nbm.FLOATS:
            assert abs(p[k] - getattr(BASE, k)) <= 0.2 * abs(getattr(BASE, k)) + 1e-4
        for k in nbm.INTS:
            assert type(p[k]) is int and p[k] >= 1
            assert abs(p[k] - getattr(BASE, k)) <= 0.2 * getattr(BASE, k) + 0.5
        assert p["tp_mode"] in nbm.TP_MODES
        Config().replace(**p)


def test_control_and_count():
    spec = nbm.generate(12, 0.2, 7, "v5")
    assert spec["variants"][0] == {"name": "V5", "group": "control", "why": "v5 as published", "params": {}}
    assert [v["name"] for v in spec["variants"][1:]] == [f"NB_{i:03d}" for i in range(1, 13)]
    assert spec["preset"] == "v5" and spec["modes"] == "c" and spec["base"] == {}


def test_summarize(tmp_path, capsys):
    rows = [("V5", "c", 10, 4, 1, 1, 1, 1)]
    rows += [(f"NB_{i}", "c", t, d, o, 1, 1, -1) for i, (t, d, o) in
             enumerate([(12, 3, 1), (8, 4, 1), (9, 5, 1), (-1, 12, -1), (7, 6, 1)])]
    rows.append(("V5", "b", 99, 1, 1, 1, 1, 1))      # other mode must be ignored
    pd.DataFrame(rows, columns=["variant", "mode", "w_totalR", "w_maxDD_R", "OOS_totalR",
                                "y2024", "y2025", "y2026"]).to_csv(tmp_path / "experiments.csv", index=False)
    assert nbm.main(["--summarize", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "neighbours: 5" in out
    assert "PASS  >=80% neighbours w_totalR>0" in out      # 4/5 = 80%
    assert "PASS  median w_totalR >= 70% of V5" in out     # 8 >= 7
    assert "PASS  >=70% OOS positive" in out               # 80%
    assert "FAIL  p95 w_maxDD_R <= 2x V5" in out           # p95 ~ 10.8 > 8
    assert "PASS  V5 below 90th percentile" in out         # rank 60%
    assert "share y2026 > 0: 0.0%" in out
