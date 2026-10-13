from vidx.checks import audio_pops, mid_word_cuts
from vidx.diff import diff_timelines, minus
from vidx.store import Word

import numpy as np


def tl(segs, **kw):
    from vidx.dsl import out_duration

    d = {"segments": [list(s) for s in segs], "speed": 1.0, "trans": None, "crop": None, "pad": 0, "snap": False}
    d.update(kw)
    d["out_duration"] = out_duration([tuple(s) for s in segs], d["speed"], d["trans"])
    return d


def test_minus():
    assert minus([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert minus([(0, 1)], [(0, 1)]) == []


def test_diff_removed_added():
    d = diff_timelines(tl([(0, 10), (20, 30)]), tl([(0, 8), (20, 35)]))
    assert d["removed"] == [(8, 10, 8)]
    assert [(s, e) for s, e, _ in d["added"]] == [(30, 35)]
    assert d["moved"] == []
    assert d["dur_b"] - d["dur_a"] == 3


def test_diff_detects_moves_minimally():
    a = tl([(0, 10), (10, 20), (20, 30)])
    b = tl([(20, 30), (0, 10), (10, 20)])  # the last block was moved to the front
    d = diff_timelines(a, b)
    assert d["removed"] == [] and d["added"] == []
    assert [(s, e) for s, e, _, _ in d["moved"]] == [(20, 30)]
    assert d["moved"][0][2:] == (20, 0)  # out 20 in a -> out 0 in b


def test_diff_options():
    d = diff_timelines(tl([(0, 10)]), tl([(0, 10)], trans=["xfade", 0.3], crop="9:16"))
    assert "trans cut -> xfade 0.3" in d["opts"] and "crop None -> 9:16" in d["opts"]


def test_identical_versions():
    d = diff_timelines(tl([(0, 10)]), tl([(0, 10)]))
    assert not (d["removed"] or d["added"] or d["moved"] or d["opts"])


def test_mid_word_cut_detection():
    words = [Word(1.0, 2.0, "alpha"), Word(3.0, 4.0, "beta")]
    hits = mid_word_cuts([(0, 1.5), (2.5, 6)], words)  # out-point inside "alpha"
    assert len(hits) == 1 and hits[0][2].text == "alpha"
    assert mid_word_cuts([(0, 2.02), (2.5, 6)], words) == []  # within 50ms tolerance


def test_audio_pop_detection():
    sr = 16000
    x = np.zeros(sr * 4, np.float32)
    x[sr * 2: sr * 3] = 0.5 * np.sin(np.arange(sr) * 2 * np.pi * 440 / sr)  # loud tone 2-3s
    pops = audio_pops([(0, 1.0), (2.5, 3.5)], x, sr)  # silence -> tone
    assert pops and pops[0][0] == 1 and pops[0][1] > 12
    assert audio_pops([(0, 1.0), (1.5, 2.0)], x, sr) == []  # silence -> silence
