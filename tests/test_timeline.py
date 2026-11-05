import numpy as np
import pytest

from vidx import dsl, render, store, timeline
from vidx.store import VidxError, VideoInfo


@pytest.fixture
def video(tmp_path, monkeypatch):
    from vidx import config

    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "vidx.db")
    monkeypatch.setattr(store, "_conn", None)
    store.put_video(VideoInfo("abc", "/tmp/x.mp4", "sha", 100.0, 30.0, 1280, 720, {"has_audio": True}))
    yield "abc"
    store.db().close()
    store._conn = None


def clips(*pairs, **kw):
    return [{"in": a, "out": b, **kw} for a, b in pairs]


def test_pieces_cover_every_frame_exactly_once():
    cs = clips((0, 2), (10, 12), (20, 21))
    cs[0]["transition"] = {"dur": 0.5}
    p = render.plan_pieces(cs, 30)
    assert [x.kind for x in p] == ["body", "join", "body", "body"]
    assert sum(x.frames for x in p) == 60 + 60 + 30 - 15  # a join overlaps its neighbours
    assert p[0].parts[0].b == pytest.approx(1.5)          # clip 1 stops where the transition starts
    assert (p[1].parts[0].a, p[1].parts[0].b) == pytest.approx((1.5, 2.0))
    assert (p[1].parts[1].a, p[1].parts[1].b) == pytest.approx((10.0, 10.5))
    assert p[2].parts[0].a == pytest.approx(10.5)         # clip 2's body starts after it


def test_speed_is_applied_to_source_time():
    p = render.plan_pieces([{"in": 10, "out": 14, "speed": 2.0}], 30)
    assert p[0].frames == 60 and (p[0].parts[0].a, p[0].parts[0].b) == pytest.approx((10, 14))


def test_validate_fills_derived_fields(video):
    tl = timeline.validate({"clips": [{"in": 1, "out": 3, "transition": {"dur": 0.5}}, {"in": 5, "out": 9, "speed": 2}]}, video)
    assert tl["segments"] == [[1.0, 3.0], [5.0, 9.0]]
    assert tl["joins"] == [0.5] and tl["speeds"] == [1.0, 2.0]
    assert tl["out_duration"] == pytest.approx(3.5)


def test_long_transition_is_clamped_with_a_warning(video):
    tl = timeline.validate({"clips": [{"in": 1, "out": 1.5, "transition": {"dur": 2}}, {"in": 5, "out": 9}]}, video)
    assert tl["joins"][0] < 0.25 and "too long" in tl["warnings"][0]


@pytest.mark.parametrize("data,needle", [
    ({"clips": []}, "non-empty"),
    ({"clips": [{"in": 3, "out": 1}]}, "clip 1: in 3.0 must be before"),
    ({"clips": [{"in": 200, "out": 300}]}, "past the end"),
    ({"clips": [{"in": 0, "out": 1, "colour": 1}]}, "unknown key 'colour'"),
    ({"clips": [{"in": 0, "out": 1, "transition": {"dur": 1}}]}, "nothing to transition into"),
    ({"clips": [{"in": 0, "out": 1}], "audio": [{"sfx": "nope"}]}, "unknown sound"),
    ({"clips": [{"in": 0, "out": 1, "captions": [{"text": "x"}]}]}, "captions need"),
])
def test_validation_errors_name_the_fix(video, data, needle):
    with pytest.raises(VidxError, match=needle):
        timeline.validate(data, video)


def test_bad_json_reports_the_line(video):
    with pytest.raises(VidxError, match="line 1"):
        timeline.load("{nope", video)


def test_dsl_compiles_to_clips(video):
    class Ctx:
        video_id, duration, fps = "abc", 60.0, 30.0
        shots = [store.Shot(1, 0, 20), store.Shot(2, 20, 60)]

        def events(self, kind):
            return []

        def words(self):
            return []

    sc = dsl.parse("drop 10-20\ntrans xfade 0.5\nmatch s2 0.5\ncaption 12 2 Hello world\nsfx impact_01 at 3 gain -9\nmusic drone_01")
    tl = dsl.resolve(sc, Ctx())
    a, b = tl["clips"]
    assert a["transition"]["dur"] == 0.5 and "transition" not in b
    assert b["grade"]["match"] == {"from": 20, "to": 60, "strength": 0.5, "video": "abc"}
    assert b["captions"] == [{"at": 2.5, "for": 2.0, "text": "Hello world"}]  # output 12 = 2.5s into clip 2
    assert tl["audio"] == [{"sfx": "impact_01", "gain": -9.0, "at": 3.0}, {"music": "drone_01", "gain": -18.0}]


def test_positions_handle_per_join_lengths_and_speeds():
    segs = [(0, 10), (20, 30), (40, 50)]
    assert dsl.out_positions(segs, [1, 2, 1], [1.0, 0.0]) == [(0, 10), (9, 14), (14, 24)]
    assert dsl.src_to_out(25, segs, [1, 2, 1], [1.0, 0.0]) == pytest.approx(11.5)


def test_colour_transfer_moves_stats_to_reference():
    from vidx import color

    rng = np.random.default_rng(0)
    lab = np.stack([rng.normal(30, 5, 5000), rng.normal(10, 2, 5000), rng.normal(-20, 3, 5000)], 1).astype(np.float32)
    src = (lab.mean(0), lab.std(0))
    ref = (np.array([60, 0, 0], np.float32), np.array([7, 3, 4], np.float32))
    out = color.transfer(lab, src, ref, strength=1.0)
    assert np.allclose(out.mean(0), ref[0], atol=0.2) and np.allclose(out.std(0), ref[1], atol=0.2)
    half = color.transfer(lab, src, ref, strength=0.5).mean(0)
    assert np.allclose(half, (src[0] + ref[0]) / 2, atol=0.2)


def test_colour_is_measured_on_what_the_viewer_sees():
    from PIL import Image

    from vidx import color

    im = Image.new("RGB", (1280, 720))
    crop = color.visible(im, [1280, 544, 0, 88], (1280, 720), 9 / 16)  # letterboxed, then 9:16
    assert abs(crop.height - 544) <= 1 and abs(crop.width - 544 * 9 / 16) <= 2


def test_sound_library_is_complete():
    names = timeline.sfx_names()
    assert len(names) == 20
    assert {n.split("_")[0] for n in names} == {"drone", "impact", "riser", "whoosh"}


def test_every_cut_has_valid_settings():
    from vidx import cuts

    for name, c in cuts.LIBRARY.items():
        assert c.kind in ("hard", "xfade", "audio"), name
        assert (c.kind == "xfade") == bool(c.xfade), name
        if c.sfx:
            assert c.sfx in timeline.sfx_names(), name
    for pack in cuts.PACKS.values():
        assert set(pack) <= set(cuts.LIBRARY) and "hard" in pack


class _Ctx:
    video_id, duration, fps = "nofeatures", 100.0, 30.0
    shots = [store.Shot(1, 0, 100)]

    def __init__(self, words=()):
        self._w = list(words)

    def words(self):
        return self._w


def test_packs_are_deterministic_and_never_repeat_an_effect(tmp_path, monkeypatch):
    from vidx import config, cuts

    monkeypatch.setattr(config, "HOME", tmp_path)
    cl = [{"in": i * 5.0, "out": i * 5.0 + 4, "video": "x"} for i in range(30)]
    a = cuts.choose(cl, "hype", _Ctx())
    assert a == cuts.choose(cl, "hype", _Ctx())          # stable, so re-renders stay cached
    assert a != cuts.choose(cl, "hype", _Ctx(), seed=9)  # a seed rerolls
    for x, y in zip(a, a[1:]):
        assert x == "hard" or x != y
    assert not any(x == y == z == "hard" for x, y, z in zip(a, a[1:], a[2:]))
    assert len(set(a)) >= 4


def test_context_steers_the_choice(tmp_path, monkeypatch):
    from vidx import config, cuts

    monkeypatch.setattr(config, "HOME", tmp_path)
    jump = cuts.JoinInfo(jump=True)
    assert "punch_in" in cuts.weigh(cuts.PACKS["comedy"], jump)
    other = cuts.weigh(cuts.PACKS["comedy"], cuts.JoinInfo())
    assert "punch_in" not in other and "j_cut" not in other  # no jump cut, no speech: nothing to hide or bridge
    talk = cuts.weigh(cuts.PACKS["clean"], cuts.JoinInfo(speech=True))
    assert talk["j_cut"] > cuts.PACKS["clean"]["j_cut"]


def test_jl_cuts_and_punches_expand_to_render_instructions():
    from vidx import cuts

    cl = [{"in": 0, "out": 4, "video": "a", "transition": {"type": "j_cut"}},
          {"in": 10, "out": 14, "video": "a", "transition": {"type": "l_cut"}},
          {"in": 20, "out": 24, "video": "a", "transition": {"type": "punch_in"}},
          {"in": 30, "out": 34, "video": "a", "transition": {"type": "punch_in"}},
          {"in": 40, "out": 44, "video": "a"}]
    out, cues = cuts.expand(cl, 30)
    assert out[0]["_tail"] == {"video": "a", "a": 9.5, "d": 0.5}   # next clip's sound, half a second early
    assert out[2]["_head"] == {"video": "a", "a": 14, "d": 0.5}    # previous clip's sound carries over
    assert out[3]["zoom"] == 1.15 and "zoom" not in out[4]          # punch in, then back out
    assert "_tail" not in cl[0]                                     # the stored timeline stays clean


def test_smash_adds_its_hit_at_the_join():
    from vidx import cuts

    _, cues = cuts.expand([{"in": 0, "out": 4, "video": "a", "transition": {"type": "smash"}},
                           {"in": 10, "out": 14, "video": "a"}], 30)
    assert cues == [{"sfx": "impact_05", "at": 4.0, "gain": -8.0}]


def test_detector_learns_a_separable_idea_and_scores_honestly():
    from vidx import detect

    rng = np.random.default_rng(5)
    concept = rng.standard_normal(64)
    concept /= np.linalg.norm(concept)
    pos = rng.standard_normal((60, 64)) * 0.05 + concept * 0.4
    neg = rng.standard_normal((180, 64)) * 0.05 - concept * 0.1
    x = np.vstack([pos, neg]).astype(np.float32)
    y = np.array([1] * 60 + [0] * 180)
    g = np.array([i // 10 for i in range(240)])
    cv = detect.cross_validate(x, y, g)
    assert cv["precision"] > 0.9 and cv["recall"] > 0.9
    noise = rng.standard_normal((240, 64)).astype(np.float32) * 0.05  # labels mean nothing here
    assert detect.cross_validate(noise, rng.permutation(y), g)["precision"] < 0.7


def test_ranges_parse_and_reject_bad_input():
    from vidx import detect

    assert detect.parse_ranges("abcd1234:1-2,abcd1234:5.5-7") == [("abcd1234", 1, 2), ("abcd1234", 5.5, 7)]
    with pytest.raises(VidxError, match="bad range"):
        detect.parse_ranges("nope")
    with pytest.raises(VidxError, match="built-in"):
        detect.check_name("laugh")
