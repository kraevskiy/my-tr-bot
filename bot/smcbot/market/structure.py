"""Swings and market structure (course L4 terminology, confirmed by the user).

* BOS  = break of structure WITH trend reversal: a candle CLOSE beyond the strong
         swing (strong low in an uptrend / strong high in a downtrend).
* Conf = confirmation / continuation: a candle CLOSE beyond the weak swing in the
         trend direction (the "repeat high" after a bullish BOS).
* Fake BOS = wick beyond the level without a close (that is a sweep).

Everything is causal: a fractal swing at bar i is only known at bar i + n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BULL, BEAR, NONE = 1, -1, 0


def fractal_swings(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Swing high at i if high[i] is the max of high[i-n..i+n] (ties: first wins).

    Returns columns sh, sl (bool at the swing bar) and the confirmation position.
    """
    h, l = df["high"].values, df["low"].values
    size = len(df)
    sh = np.zeros(size, bool)
    sl = np.zeros(size, bool)
    for i in range(n, size - n):
        wh = h[i - n:i + n + 1]
        wl = l[i - n:i + n + 1]
        if h[i] == wh.max() and np.argmax(wh) == n:
            sh[i] = True
        if l[i] == wl.min() and np.argmin(wl) == n:
            sl[i] = True
    return pd.DataFrame({"sh": sh, "sl": sl}, index=df.index)


@dataclass
class StructureEvent:
    pos: int            # bar position where the event happened (close of that bar)
    kind: str           # BOS_UP, BOS_DOWN, CONF_UP, CONF_DOWN
    level: float        # level that was broken
    first_conf: bool = False


def market_structure(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, list[StructureEvent]]:
    """Run the BOS/Conf state machine. Output row i = state after bar i closed."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    sw = fractal_swings(df, n)
    sh_flags, sl_flags = sw["sh"].values, sw["sl"].values
    size = len(df)

    trend = NONE
    strong = np.nan        # strong low (bull) / strong high (bear)
    strong_pos = -1
    weak = np.nan          # weak high (bull) / weak low (bear): level for Conf
    weak_pos = -1
    pending_since = 0      # start of the current leg (after last event)
    conf_since_bos = 0
    last_sh = (np.nan, -1)
    last_sl = (np.nan, -1)

    out_trend = np.zeros(size, int)
    out_strong = np.full(size, np.nan)
    out_weak = np.full(size, np.nan)
    out_event = np.array([""] * size, dtype=object)
    events: list[StructureEvent] = []

    for i in range(size):
        # swings confirmed at this bar (formed at i - n)
        j = i - n
        if j >= 0:
            if sh_flags[j]:
                last_sh = (h[j], j)
                if trend == BULL and j >= pending_since and (np.isnan(weak) or weak_pos < pending_since):
                    # first pullback after the event fixes the weak high of the leg
                    seg = h[pending_since:j + 1]
                    weak = seg.max()
                    weak_pos = pending_since + int(np.argmax(seg))
            if sl_flags[j]:
                last_sl = (l[j], j)
                if trend == BEAR and j >= pending_since and (np.isnan(weak) or weak_pos < pending_since):
                    seg = l[pending_since:j + 1]
                    weak = seg.min()
                    weak_pos = pending_since + int(np.argmin(seg))

        if trend == NONE:
            # bootstrap from the first confirmed swings
            if not np.isnan(last_sh[0]) and c[i] > last_sh[0] and not np.isnan(last_sl[0]):
                trend, strong, strong_pos = BULL, last_sl[0], last_sl[1]
                pending_since, weak, weak_pos, conf_since_bos = i, np.nan, -1, 0
                events.append(StructureEvent(i, "BOS_UP", last_sh[0]))
                out_event[i] = "BOS_UP"
            elif not np.isnan(last_sl[0]) and c[i] < last_sl[0] and not np.isnan(last_sh[0]):
                trend, strong, strong_pos = BEAR, last_sh[0], last_sh[1]
                pending_since, weak, weak_pos, conf_since_bos = i, np.nan, -1, 0
                events.append(StructureEvent(i, "BOS_DOWN", last_sl[0]))
                out_event[i] = "BOS_DOWN"
        elif trend == BULL:
            if c[i] < strong:
                lvl = strong
                seg_start = max(strong_pos, 0)
                seg = h[seg_start:i + 1]
                trend = BEAR
                strong = seg.max()
                strong_pos = seg_start + int(np.argmax(seg))
                pending_since, weak, weak_pos, conf_since_bos = i, np.nan, -1, 0
                events.append(StructureEvent(i, "BOS_DOWN", lvl))
                out_event[i] = "BOS_DOWN"
            elif not np.isnan(weak) and c[i] > weak:
                lvl = weak
                seg = l[weak_pos:i + 1]
                strong = seg.min()
                strong_pos = weak_pos + int(np.argmin(seg))
                conf_since_bos += 1
                events.append(StructureEvent(i, "CONF_UP", lvl, conf_since_bos == 1))
                out_event[i] = "CONF_UP"
                pending_since, weak, weak_pos = i, np.nan, -1
        else:  # BEAR
            if c[i] > strong:
                lvl = strong
                seg_start = max(strong_pos, 0)
                seg = l[seg_start:i + 1]
                trend = BULL
                strong = seg.min()
                strong_pos = seg_start + int(np.argmin(seg))
                pending_since, weak, weak_pos, conf_since_bos = i, np.nan, -1, 0
                events.append(StructureEvent(i, "BOS_UP", lvl))
                out_event[i] = "BOS_UP"
            elif not np.isnan(weak) and c[i] < weak:
                lvl = weak
                seg = h[weak_pos:i + 1]
                strong = seg.max()
                strong_pos = weak_pos + int(np.argmax(seg))
                conf_since_bos += 1
                events.append(StructureEvent(i, "CONF_DOWN", lvl, conf_since_bos == 1))
                out_event[i] = "CONF_DOWN"
                pending_since, weak, weak_pos = i, np.nan, -1

        out_trend[i] = trend
        out_strong[i] = strong
        out_weak[i] = weak

    state = pd.DataFrame({"trend": out_trend, "strong": out_strong, "weak": out_weak,
                          "event": out_event}, index=df.index)
    return state, events


def swing_levels(df: pd.DataFrame, n: int, bar: pd.Timedelta) -> pd.DataFrame:
    """List of swings with the time they become known (close of bar p + n)."""
    sw = fractal_swings(df, n)
    rows = []
    idx = df.index
    for kind, col, price in (("high", "sh", "high"), ("low", "sl", "low")):
        pos = np.flatnonzero(sw[col].values)
        for p in pos:
            if p + n < len(df):
                rows.append((idx[p], idx[p + n] + bar, kind, df[price].iat[p], p))
    out = pd.DataFrame(rows, columns=["time", "known_at", "kind", "price", "pos"])
    return out.sort_values("known_at").reset_index(drop=True)
