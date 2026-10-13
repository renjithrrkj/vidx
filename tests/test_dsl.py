import pytest

from vidx import dsl
from vidx.store import Event, Shot, VidxError, Word


class FakeCtx:
    """Stand-in for dsl.IndexCtx: synthetic 60s video, no DB."""

    def __init__(self, duration=60.0, shots=None, events=None, words=None):
        self.video_id, self.duration, self.fps = "fake", duration, 30.0
        self.shots = shots or [Shot(1, 0, 20), Shot(2, 20, 40), Shot(3, 40, 60)]
        self._events = events or []
        self._words = words or []

    def events(self, kind):
        return [e for e in self._events if e.kind == kind]

    def words(self):
        return self._words


def resolve(text, **kw):
    return dsl.resolve(dsl.parse(text), FakeCtx(**kw))


# ---------------------------------------------------------------- parser

def test_parse_defaults_and_comments():
    sc = dsl.parse("# just a comment\n\nbase all  # trailing\npad 0.15\nsnap words\ntrans xfade 0.3\n"
                   "crop 9:16 center\nspeed 1.25")
    assert sc.base == "all" and sc.pad == 0.15 and sc.snap
    assert sc.trans == ("xfade", 0.3) and sc.crop == "9:16" and sc.speed == 1.25


@pytest.mark.parametrize("text,needle", [
    ("drop 1-2\nfrobnicate", "line 2: unknown command"),
    ("base sometimes", "line 1: base takes"),
    ("base all\nbase none", "line 2: base given twice"),
    ("pad lots", "line 1: bad time"),
    ("trans wipe 0.3", "line 1: trans takes"),
    ("speed 9", "line 1: speed must be"),
    ("crop 9:16 face", "face-tracked crop is v2"),
    ("drop", "line 1: drop takes one target"),
])
def test_parse_errors_cite_line(text, needle):
    with pytest.raises(VidxError, match=needle):
        dsl.parse(text)


def test_times_accept_mmss():
    tl = resolve("drop 0:10-1:00.5")  # 10s to 60.5 -> clamps at 60
    assert tl["segments"] == [[0.0, 10.0]]


# ---------------------------------------------------------------- segment math

def test_subtract_splits_and_trims():
    assert dsl.subtract([(0, 10)], 2, 3) == [(0, 2), (3, 10)]
    assert dsl.subtract([(0, 10)], -1, 3) == [(3, 10)]
    assert dsl.subtract([(0, 5), (6, 10)], 4, 7) == [(0, 4), (7, 10)]
    assert dsl.subtract([(0, 5)], 6, 7) == [(0, 5)]


def test_union_sorted_merges():
    assert dsl.union_sorted([(0, 2), (5, 6)], 1, 5.5) == [(0, 6)]


def test_merge_close_only_forward():
    assert dsl.merge_close([(0, 1), (1.05, 2)]) == [(0, 2)]
    assert dsl.merge_close([(0, 1), (1.2, 2)]) == [(0, 1), (1.2, 2)]
    # going backwards in source time (a reorder) is never merged
    assert dsl.merge_close([(5, 6), (0, 1)]) == [(5, 6), (0, 1)]


def test_drop_short():
    keep, gone = dsl.drop_short([(0, 1), (2, 2.2), (3, 3.3)])
    assert keep == [(0, 1), (3, 3.3)] and gone == [(2, 2.2)]


def test_base_all_drops():
    tl = resolve("drop 10-20\ndrop s3")
    assert tl["segments"] == [[0.0, 10.0], [20.0, 40.0]]
    assert tl["out_duration"] == 30.0


def test_base_none_keeps_in_line_order():
    tl = resolve("base none\nkeep s3\nkeep 5-10\nkeep s1")
    # s1 overlaps 5-10 but comes later: reorders are allowed, merges only go forward
    assert tl["segments"] == [[40.0, 60.0], [5.0, 10.0], [0.0, 20.0]]


def test_base_none_without_keep_errors():
    with pytest.raises(VidxError, match="needs at least one keep"):
        resolve("base none\ndrop 1-2")


def test_drop_events_and_shot_ranges():
    ev = [Event("silence", 5, 6.5, 1.5), Event("silence", 30, 30.8, 0.8)]
    tl = resolve("drop silence>1.0\ndrop s2-s3", events=ev)
    assert tl["segments"] == [[0.0, 5.0], [6.5, 20.0]]


def test_unknown_shot_error():
    with pytest.raises(VidxError, match=r"line 1: unknown shot s99 \(video has s1-s3\)"):
        resolve("drop s99")


def test_short_segments_discarded_with_warning():
    with pytest.raises(VidxError, match="nothing to render"):
        resolve("drop 0-10\ndrop 10.2-60")
    tl = resolve("drop 10-20\ndrop 20.2-30")
    assert [20.0, 20.2] not in tl["segments"]
    assert any("shorter than 0.3s" in w for w in tl["warnings"])


def test_drop_matching_nothing_warns():
    tl = resolve("drop black")
    assert any("matched nothing" in w for w in tl["warnings"])


# ---------------------------------------------------------------- snap / pad

WORDS = [Word(1.0, 1.5, "hello"), Word(1.6, 2.4, "there"), Word(3.0, 3.6, "general"), Word(3.7, 4.2, "kenobi")]


def test_snap_point_moves_into_nearest_gap():
    # 2.3 is inside "there" (1.6-2.4) near its end -> the gap after it (2.4-3.0)
    assert 2.4 <= dsl.snap_point(2.3, "end", WORDS, 10) <= 3.0
    # 1.65 is near the start of "there" -> gap before it (1.5-1.6)
    assert 1.5 <= dsl.snap_point(1.65, "start", WORDS, 10) <= 1.6
    # outside any word: untouched
    assert dsl.snap_point(2.7, "end", WORDS, 10) == 2.7


def test_snap_roles_hug_the_kept_word():
    # end cut in a long gap sits just after the previous word; start cut just before the next
    words = [Word(1, 2, "a"), Word(6, 7, "b")]
    assert dsl.snap_point(1.9, "end", words, 10) == pytest.approx(2.15)
    assert dsl.snap_point(6.1, "start", words, 10) == pytest.approx(5.85)


def test_snap_segments_end_to_end():
    # drop 2.0-3.3 cuts through "there" and "general"
    tl = resolve("drop 2.0-3.3\nsnap words", words=WORDS, duration=10)
    (a0, a1), (b0, b1) = tl["segments"]
    for t in (a1, b0):
        assert not any(w.start + 0.02 < t < w.end - 0.02 for w in WORDS)


def test_pad_does_not_enter_neighbouring_words():
    segs = dsl.pad_segments([(3.0, 3.6)], 0.5, 10, WORDS)
    s, e = segs[0]
    assert s >= 2.4 and e <= 3.7  # stops at the end of "there" / start of "kenobi"
    assert dsl.pad_segments([(5, 6)], 0.2, 10) == [(4.8, 6.2)]
    assert dsl.pad_segments([(0.1, 9.9)], 0.5, 10) == [(0.0, 10.0)]


def test_snap_then_pad_then_merge():
    ev = [Event("silence", 10, 10.15, 0.15)]
    tl = resolve("drop 10-10.15\npad 0.1", events=ev)
    assert tl["segments"] == [[0.0, 60.0]]  # padding closes the gap -> merged back


# ---------------------------------------------------------------- transitions / positions

def test_xfade_out_duration_and_clamp():
    tl = resolve("drop 10-20\ntrans xfade 0.5")
    assert tl["out_duration"] == pytest.approx(49.5)
    tl = resolve("base none\nkeep 0-0.5\nkeep 10-20\ntrans xfade 0.5")
    assert tl["trans"][1] < 0.25 and any("xfade" in w for w in tl["warnings"])


def test_out_positions_and_mapping():
    segs = [(0, 10), (20, 30)]
    assert dsl.out_positions(segs) == [(0, 10), (10, 20)]
    assert dsl.out_positions(segs, trans=("xfade", 1.0)) == [(0, 10), (9, 19)]
    assert dsl.out_positions(segs, speed=2) == [(0, 5), (5, 10)]
    assert dsl.src_to_out(25, segs) == 15
    assert dsl.out_to_src(15, segs) == 25
    assert dsl.src_to_out(15, segs) is None


def test_acoustic_words_trims_phantom_timing():
    words = [Word(1.0, 2.0, "a"), Word(2.5, 3.3, "phantom"), Word(3.0, 4.0, "b")]
    sil = [(2.2, 3.2)]
    out = dsl.acoustic_words(words, sil)
    assert [(w.text, w.start, w.end) for w in out] == [("a", 1.0, 2.0), ("phantom", 3.2, 3.3), ("b", 3.2, 4.0)]
    assert [w.text for w in dsl.acoustic_words([Word(2.3, 2.9, "ghost")], sil)] == []
