"""Stills and image layers, checked on real renders: what matters is which pixels end up where."""
import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from vidx import api, dsl, render, store, timeline
from vidx.store import VideoInfo, VidxError

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
RED, BLUE = (255, 0, 0), (0, 0, 255)


@pytest.fixture
def home(tmp_path, monkeypatch):
    from vidx import config

    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "vidx.db")
    monkeypatch.setattr(store, "_conn", None)
    monkeypatch.setenv("VIDX_OUTPUT_DIR", str(tmp_path / "out"))
    yield tmp_path
    store.db().close()
    store._conn = None


@pytest.fixture
def video(home):
    """12s of mid grey with a tone, registered without running the analyzers."""
    p = home / "grey.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=30:d=12",
                    "-f", "lavfi", "-i", "sine=f=440:d=12", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(p)], check=True)
    store.put_video(VideoInfo("grey", str(p), "sha", 12.0, 30.0, 640, 360, {"has_audio": True}))
    return "grey"


@pytest.fixture
def fake(home):
    store.put_video(VideoInfo("abc", "/tmp/x.mp4", "sha", 100.0, 30.0, 1280, 720, {"has_audio": True}))
    return "abc"


def picture(path, size=(400, 300), color=RED, mode="RGB", **save):
    Image.new(mode, size, color).save(path, **save)
    return str(path)


def frame(path, t):
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    info = render.util.ffprobe(str(path))
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    return np.frombuffer(p.stdout, np.uint8).reshape(vs["height"], vs["width"], 3).astype(int)


def is_red(px):
    return px[0] > 180 and px[1] < 90 and px[2] < 90


def render_of(reply, **kw):
    ver = reply.split(":")[0].split()[0]
    api.render(ver, **kw)
    return ver, store.get_version(ver)["output_path"]


# ---------------------------------------------------------------- preparing images


def test_phone_photo_is_stood_upright(home, tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, as phones do
    src = picture(tmp_path / "phone.jpg", (400, 200), exif=exif)
    _, _, w, h = timeline.prepare_image(src)
    assert (w, h) == (200, 400)


def test_transparency_survives_and_huge_images_are_capped(home, tmp_path):
    _, p, *_ = timeline.prepare_image(picture(tmp_path / "logo.png", color=(255, 0, 0, 0), mode="RGBA"))
    assert Image.open(p).mode == "RGBA"
    _, _, w, h = timeline.prepare_image(picture(tmp_path / "big.png", (8000, 4000)))
    assert (w, h) == (timeline.MAX_IMAGE_SIDE, timeline.MAX_IMAGE_SIDE // 2)


def test_edit_survives_the_original_being_moved(home, tmp_path):
    src = tmp_path / "a.png"
    picture(src)
    asset = timeline.prepare_image(str(src))[0]
    src.unlink()
    assert timeline.prepare_image(str(src), asset)[0] == asset
    with pytest.raises(VidxError, match="not found"):
        timeline.prepare_image(str(src))


def test_not_an_image_is_named_clearly(home, tmp_path):
    bad = tmp_path / "notes.jpg"
    bad.write_text("hello")
    with pytest.raises(VidxError, match="notes.jpg"):
        timeline.prepare_image(str(bad))


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("ov,msg", [
    ({"pos": "middle"}, "unknown pos"), ({"pos": [2, 0]}, "pos is a name"), ({"width": 0}, "width"), ({"width": 1.5}, "width"),
    ({"opacity": 0}, "opacity"), ({"fade": 3, "for": 2}, "fade"), ({"motion": "spin"}, "motion"), ({"for": -1}, "for"),
    ({"colour": 1}, "unknown overlay key")])
def test_bad_overlays_are_refused(fake, tmp_path, ov, msg):
    tl = {"clips": [{"in": 0, "out": 5, "overlays": [{"image": picture(tmp_path / "a.png"), **ov}]}]}
    with pytest.raises(VidxError, match=msg):
        timeline.validate(tl, fake)


@pytest.mark.parametrize("c,msg", [({"motion": "spin"}, "motion"), ({"fit": "stretch"}, "fit"), ({"for": 0.01}, "for"),
                                   ({"for": 9999}, "for"), ({"crop": "9:16"}, "crop")])
def test_bad_stills_are_refused(fake, tmp_path, c, msg):
    with pytest.raises(VidxError, match=msg):
        timeline.validate({"clips": [{"image": picture(tmp_path / "a.png"), **c}]}, fake)


def test_a_timeline_can_be_read_back_after_export(fake, tmp_path):
    src = picture(tmp_path / "a.png")
    tl = timeline.validate({"clips": [{"in": 0, "out": 5, "overlays": [{"image": src}]}, {"image": src, "for": 2}]}, fake)
    again = timeline.validate(json.loads(json.dumps(timeline.export(tl))), fake)
    assert again["clips"] == tl["clips"] and again["out_duration"] == 7


def test_stills_stay_out_of_the_video_list(fake, tmp_path):
    timeline.validate({"clips": [{"image": picture(tmp_path / "a.png"), "for": 2}]}, fake)
    assert [v.id for v in store.list_videos()] == ["abc"]


# ---------------------------------------------------------------- DSL


def edit(script, vid="abc"):
    return dsl.resolve(dsl.parse(script), dsl.IndexCtx(vid))


def test_still_splits_the_clip_it_lands_in(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    c = edit(f"base none\nkeep 10-20\nstill {a} 3 at 4")["clips"]
    assert [(x["in"], x["out"]) for x in c] == [(10, 14), (0, 3), (14, 20)] and c[1]["image"] == a


def test_still_times_ignore_other_stills_and_order_is_kept(fake, tmp_path):
    a, b, d = (picture(tmp_path / f"{n}.png", color=col) for n, col in (("a", RED), ("b", BLUE), ("d", (0, 255, 0))))
    c = edit(f"base none\nkeep 0-10\nstill {a} 1 at 2\nstill {b} 1 at 6\nstill {d} 1 at 6")["clips"]
    assert [x.get("image", (x["in"], x["out"])) for x in c] == [(0, 2), a, (2, 6), b, d, (6, 10)]


def test_still_at_the_edges_does_not_leave_slivers(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    assert edit(f"base none\nkeep 0-5\nstill {a} 1 at 0")["clips"][0].get("image") == a
    assert edit(f"base none\nkeep 0-5\nstill {a} 1 at end")["clips"][-1].get("image") == a
    assert len(edit(f"base none\nkeep 0-5\nstill {a} 1 at 4.95")["clips"]) == 2


def test_still_respects_speed(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    c = edit(f"base none\nkeep 10-20\nspeed 2\nstill {a} 1 at 2")["clips"]
    assert c[0]["out"] == 14 and c[2]["in"] == 14  # 2s of output is 4s of source


def test_every_join_gets_the_global_transition(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    c = edit(f"base none\nkeep 0-10\ntrans xfade 0.3\nstill {a} 2 at 5")["clips"]
    assert [bool(x.get("transition")) for x in c] == [True, True, False]


def test_cut_packs_work_across_stills(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    tl = edit(f"base none\nkeep 0-10\nkeep 20-30\nstill {a} 2 at 5\ncuts hype 3")
    assert len(tl["clips"]) == 4 and tl["out_duration"] > 0


def test_paths_with_spaces(fake, tmp_path):
    a = picture(tmp_path / "my photo.png")
    assert edit(f'base none\nkeep 0-5\nstill "{a}" 1 at end')["clips"][-1]["image"] == a
    with pytest.raises(VidxError, match="unmatched quote"):
        dsl.parse(f'still "{a} 1')


@pytest.mark.parametrize("line", ["still a.png", "still a.png 2 at", "still a.png 2 spin 1", "overlay a.png 1",
                                  "overlay a.png 1 2 size 3", "overlay a.png 1 2 width big", "overlay a.png 1 2 pos a,b"])
def test_bad_lines_say_how_to_write_them(line):
    with pytest.raises(VidxError):
        dsl.parse(line)


def test_overlay_is_copied_to_each_clip_it_touches(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    c = edit(f"base none\nkeep 0-4\nkeep 10-14\nkeep 20-24\noverlay {a} 2 7 fade 0.5")["clips"]
    assert [o["at"] for x in c for o in x.get("overlays", [])] == [2, -2, -6]  # one layer: 2-9 of the edit
    assert all(o["for"] == 7 for x in c for o in x["overlays"])
    assert "overlays" not in edit(f"base none\nkeep 0-4\nkeep 10-14\nkeep 20-24\noverlay {a} 2 3")["clips"][2]


def test_overlay_all_runs_to_the_end_and_past_the_end_is_refused(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    c = edit(f"base none\nkeep 0-4\nkeep 10-14\noverlay {a} 1 all")["clips"]
    assert c[1]["overlays"][0]["for"] == 7
    with pytest.raises(VidxError, match="edit ends"):
        edit(f"base none\nkeep 0-4\noverlay {a} 9 1")


def test_captions_land_after_stills_move_things(fake, tmp_path):
    a = picture(tmp_path / "a.png")
    c = edit(f"base none\nkeep 0-10\nstill {a} 2 at 5\ncaption 8 1 Hi")["clips"]
    assert c[2]["captions"][0]["at"] == 1  # 8 of the output is 1s into the clip after the still


# ---------------------------------------------------------------- layer maths


@pytest.mark.parametrize("pos,xy", [("top_left", (29, 29)), ("top_right", (1920 - 480 - 43, 43)), ("center", (720, 450)),
                                    ("bottom", (720, 1080 - 180 - 43)), ([0.5, 0.25], (960, 270))])
def test_overlay_box(pos, xy):
    w, h, x, y = render.overlay_box({"width": 0.25, "pos": pos}, 1920, 1080, 800, 300)
    assert (w, h) == (480, 180)
    assert (x, y) == ((43, 43) if pos == "top_left" else xy)


def test_window_of_a_layer_inside_a_piece():
    assert render.overlay_window(1, 2, 0.5, 5) == (1, 2, True, 1.5)
    assert render.overlay_window(1, 2, 0, 5) == (1, 2, False, None)


def test_window_of_a_long_layer_is_only_what_the_piece_needs():
    start, span, fade_in, fade_out = render.overlay_window(-300, 600, 0.5, 4)
    assert (start, fade_in, fade_out) == (0, False, None) and span == pytest.approx(4.1)


def test_window_keeps_fades_that_cross_a_piece_boundary():
    assert render.overlay_window(-0.2, 2, 0.5, 5)[:3] == (-0.2, 2, True)  # fade-in began in the last piece
    start, span, fade_in, fade_out = render.overlay_window(4, 2, 0.5, 5)
    assert fade_in and fade_out is None                                    # fade-out belongs to the next piece
    assert render.overlay_window(-1.8, 2, 0.5, 5)[3] == 0                  # already fading out


def test_pieces_of_a_moving_still_share_one_motion(home, tmp_path):
    spec = render.Spec(640, 360, 30.0, "original", False, tmp_path)
    c = {"in": 0, "out": 3, "motion": "zoom_in", "speed": 1.0}
    assert "on+0" in render.still_chain(c, render.Part(0, 0, 2, 60, 0.0), spec)
    assert "on+60" in render.still_chain(c, render.Part(0, 2, 3, 30, 2.0), spec)


# ---------------------------------------------------------------- real renders


@needs_ffmpeg
def test_layer_shows_where_and_when_asked(video, tmp_path):
    a = picture(tmp_path / "logo.png")
    _, out = render_of(api.edit(video, f"overlay {a} 3 2 pos top_right width 0.25"))
    assert not is_red(frame(out, 2.5)[40, 560])
    assert is_red(frame(out, 4)[40, 560]) and not is_red(frame(out, 4)[320, 60])
    assert not is_red(frame(out, 5.5)[40, 560])


@needs_ffmpeg
def test_transparent_parts_stay_see_through(video, tmp_path):
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    im.paste((255, 0, 0, 255), (0, 0, 100, 200))
    im.save(tmp_path / "half.png")
    _, out = render_of(api.edit(video, f"overlay {tmp_path / 'half.png'} 0 all pos center width 0.5"))
    f = frame(out, 2)
    assert is_red(f[180, 240]) and abs(f[180, 400] - f[10, 10]).max() < 12


@needs_ffmpeg
def test_opacity_and_fade_change_strength(video, tmp_path):
    a = picture(tmp_path / "logo.png")
    _, out = render_of(api.edit(video, f"overlay {a} 2 4 pos center width 0.4 opacity 0.5 fade 1"))
    red = lambda t: frame(out, t)[180, 320][0] - frame(out, t)[180, 320][2]
    assert red(1.5) < 15 and 25 < red(2.5) < red(4) and 80 < red(4) < 180 and red(5.5) < red(4)


@needs_ffmpeg
def test_layer_slides_in(video, tmp_path):
    a = picture(tmp_path / "logo.png")
    _, out = render_of(api.edit(video, f"overlay {a} 2 3 pos left width 0.25 motion slide_left"))
    assert not is_red(frame(out, 2.05)[180, 80])
    assert is_red(frame(out, 3)[180, 80])


@needs_ffmpeg
def test_layer_carries_across_a_cut_and_a_transition(video, tmp_path):
    a = picture(tmp_path / "logo.png")
    _, out = render_of(api.edit(video, f"base none\nkeep 0-3\nkeep 5-8\nkeep 9-12\ncut 1 dissolve\noverlay {a} 1 all pos top_left width 0.2"))
    for t in (1.5, 2.8, 3.0, 5.2, 7.5):
        assert is_red(frame(out, t)[30, 40]), t


@needs_ffmpeg
def test_still_replaces_the_picture_for_its_length(video, tmp_path):
    a = picture(tmp_path / "photo.png", (900, 500))
    ver, out = render_of(api.edit(video, f"base none\nkeep 0-4\nstill {a} 2 at 2"))
    info = render.util.ffprobe(out)
    assert float(info["format"]["duration"]) == pytest.approx(6, abs=0.1)
    assert not is_red(frame(out, 1)[180, 320]) and is_red(frame(out, 3)[180, 320]) and not is_red(frame(out, 5)[180, 320])
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (640, 360)  # the still doesn't resize the edit


@needs_ffmpeg
def test_still_first_keeps_the_videos_size(video, tmp_path):
    a = picture(tmp_path / "photo.png", (1000, 1000))
    _, out = render_of(api.edit(video, f"base none\nkeep 0-3\nstill {a} 2 at 0"))
    v = next(s for s in render.util.ffprobe(out)["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (640, 360) and is_red(frame(out, 1)[10, 10])


@needs_ffmpeg
def test_contain_shows_the_whole_picture_with_bars(video, tmp_path):
    a = picture(tmp_path / "tall.png", (300, 900))
    _, out = render_of(api.edit(video, f"base none\nkeep 0-3\nstill {a} 2 at 0 fit contain"))
    f = frame(out, 1)
    assert is_red(f[180, 320]) and f[180, 40].max() < 30


def gradient(path):
    x = np.tile(np.linspace(0, 255, 1600), (900, 1)).astype(np.uint8)
    Image.fromarray(np.stack([x, x, x], -1)).save(path)
    return str(path)


@needs_ffmpeg
@pytest.mark.parametrize("motion,moves", [("zoom_in", True), ("pan_left", True), ("none", False)])
def test_motion_moves_the_picture(video, tmp_path, motion, moves):
    a = gradient(tmp_path / "g.png")
    _, out = render_of(api.edit(video, f"base none\nkeep 0-2\nstill {a} 3 at 0 motion {motion}"))
    change = abs(frame(out, 0.1) - frame(out, 2.8)).mean()
    assert (change > 3) == moves, change


@needs_ffmpeg
def test_a_slow_move_does_not_shake(video, tmp_path):
    """Track an edge frame by frame: it must always travel one way."""
    im = Image.new("RGB", (1600, 900), (0, 0, 0))
    im.paste((255, 255, 255), (0, 0, 800, 900))
    im.save(tmp_path / "edge.png")
    _, out = render_of(api.edit(video, f"base none\nkeep 0-2\nstill {tmp_path / 'edge.png'} 4 at 0 motion pan_left"))
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", "3.9", "-i", out, "-vf", "crop=iw:2:0:180",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    rows = np.frombuffer(p.stdout, np.uint8).reshape(-1, 2, 640)[:, 0].astype(float)
    edge = (rows / 255).sum(1)  # sub-pixel edge position
    steps = np.diff(edge)
    direction = np.sign(np.median(steps))
    assert direction != 0
    assert (steps * direction > -0.25).all(), steps.round(2)  # never jumps back


@needs_ffmpeg
def test_motion_continues_through_a_transition(video, tmp_path):
    a = gradient(tmp_path / "g.png")
    ver, out = render_of(api.edit(video, f"base none\nkeep 0-2\nkeep 4-6\nstill {a} 3 at 2 motion zoom_in\ncut 1 dissolve\ncut 2 dissolve"))
    assert float(render.util.ffprobe(out)["format"]["duration"]) == pytest.approx(store.get_version(ver)["timeline"]["out_duration"], abs=0.15)


@needs_ffmpeg
def test_changing_one_layer_only_reencodes_its_clip(video, tmp_path):
    a = picture(tmp_path / "logo.png")
    base = "base none\nkeep 0-3\nkeep 5-8\n"
    v1 = api.edit(video, base + f"overlay {a} 0.5 1").split(":")[0].split()[0]
    assert "2 encoded" in api.render(v1)
    v2 = api.edit(video, base + f"overlay {a} 0.5 1 pos left").split(":")[0].split()[0]
    assert "1 encoded" in api.render(v2)
    v3 = api.edit(video, base + f"overlay {a} 0.5 1").split(":")[0].split()[0]
    assert "0 encoded" in api.render(v3)


@needs_ffmpeg
def test_swapping_the_picture_is_not_served_from_the_cache(video, tmp_path):
    src = tmp_path / "logo.png"
    picture(src, color=RED)
    _, one = render_of(api.edit(video, f"overlay {src} 0 all pos center"))
    picture(src, color=BLUE)
    _, two = render_of(api.edit(video, f"overlay {src} 0 all pos center"))
    assert one != two and not is_red(frame(two, 1)[180, 320])


@needs_ffmpeg
def test_check_does_not_call_a_still_a_freeze(video, tmp_path):
    a = picture(tmp_path / "photo.png")
    ver, _ = render_of(api.edit(video, f"base none\nkeep 0-4\nstill {a} 4 at 2"))
    # the grey test clip is itself a freeze at 0-2 and 6-8; only the still's 2-6 must stay quiet
    assert "freeze 2." not in api.check(ver) and "freeze 3." not in api.check(ver)


@needs_ffmpeg
def test_diff_names_media_and_ignores_still_times(video, tmp_path):
    a = picture(tmp_path / "photo.png")
    v1 = api.edit(video, "base none\nkeep 5-8").split(":")[0].split()[0]
    v2 = api.edit(video, f"base none\nkeep 5-8\nstill {a} 2 at end\noverlay {a} 0 1").split(":")[0].split()[0]
    d = api.diff(v1, v2)
    assert "added still photo.png 2s" in d and "added overlay photo.png" in d and "added src" not in d


@needs_ffmpeg
def test_edit_reports_stills_and_layers(video, tmp_path):
    a = picture(tmp_path / "photo.png")
    r = api.edit(video, f"base none\nkeep 0-3\nstill {a} 2 at end\noverlay {a} 0 1")
    assert "1 still(s)" in r and "1 overlay image(s)" in r and "still 2s" in r


@needs_ffmpeg
def test_stills_are_silent_and_the_reel_preset_fits_them(video, tmp_path):
    a = picture(tmp_path / "photo.png", (900, 500))
    _, out = render_of(api.edit(video, f"base none\nkeep 0-3\nstill {a} 2 at end motion zoom_out\nmusic drone_01 gain -30"), preset="reel_9x16", preview=True)
    v = next(s for s in render.util.ffprobe(out)["streams"] if s["codec_type"] == "video")
    assert v["height"] > v["width"]
