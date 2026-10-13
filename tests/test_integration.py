"""End-to-end on a generated clip: index -> query -> edit -> render -> check -> diff.

Clip layout (20s, 640x360):
  0-6   testsrc2 + 440Hz        6-8  testsrc2 + SILENCE
  8-14  smptebars + 660Hz       14-15 BLACK + silence
  15-20 testsrc + 880Hz
Needs ffmpeg plus the whisper/CLIP weights (downloaded on first run). Skip with -m "not slow".
"""
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.slow


def make_clip(path):
    v = ("testsrc2=s=640x360:r=30:d=8[a];smptebars=s=640x360:r=30:d=6[b];"
         "color=c=black:s=640x360:r=30:d=1[c];testsrc=s=640x360:r=30:d=5[d];"
         "[a][b][c][d]concat=n=4:v=1:a=0,format=yuv420p[v];"
         "sine=f=440:d=6[t1];anullsrc=r=44100:cl=mono,atrim=0:2[s1];sine=f=660:d=6[t2];"
         "anullsrc=r=44100:cl=mono,atrim=0:1[s2];sine=f=880:d=5[t3];"
         "[t1]aresample=44100[u1];[t2]aresample=44100[u2];[t3]aresample=44100[u3];"
         "[u1][s1][u2][s2][u3]concat=n=5:v=0:a=1,volume=0.3[out]")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-filter_complex", v, "-map", "[v]", "-map", "[out]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)], check=True)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    from vidx import config, store

    home = tmp_path_factory.mktemp("vidxhome")
    old = (config.HOME, config.DB_PATH, store._conn)
    config.HOME, config.DB_PATH, store._conn = home, home / "vidx.db", None
    clip = home / "clip.mp4"
    make_clip(clip)
    yield clip
    config.HOME, config.DB_PATH, store._conn = old


def test_end_to_end(env):
    from vidx import api, store, util

    out = api.index_video(str(env))
    vid = out.split()[1]
    v = store.get_video(vid)
    assert abs(v.duration - 20) < 0.2

    shots = store.get_shots(vid)
    cuts = [s.start for s in shots[1:]]
    for expected in (8.0, 15.0):
        assert any(abs(c - expected) < 0.2 for c in cuts), cuts

    sil = store.get_events(vid, "silence")
    assert any(abs(e.start - 6) < 0.2 and abs(e.end - 8) < 0.2 for e in sil), sil
    blacks = store.get_events(vid, "black")
    assert any(abs(e.start - 14) < 0.2 for e in blacks), blacks

    # re-index is a no-op
    again = api.index_video(str(env))
    assert again.split()[1] == vid

    q = api.query(vid, "silence>1.5")
    assert q.startswith("1 silence") or q.startswith("2 silence"), q
    ov = api.overview(vid)
    assert len(ov) < 12000 and "c1" in ov

    imgs, cap = api.peek(vid, ["1", "s2"])
    assert len(imgs) == 1 and imgs[0][:2] == b"\xff\xd8"

    e1 = api.edit(vid, "drop silence>1.0\ndrop black\npad 0.1")
    assert e1.startswith("v")
    ver1 = e1.split(":")[0].split()[0]
    r = api.render(ver1)
    info = util.ffprobe(store.get_version(ver1)["output_path"])
    dur = float(info["format"]["duration"])
    expect = store.get_version(ver1)["timeline"]["out_duration"]
    assert abs(dur - expect) < 0.15, (dur, expect, r)

    rep = api.check(ver1)
    assert "black" not in rep.split("\n", 1)[-1] or "from source" in rep, rep

    e2 = api.edit(vid, "trans xfade 0.3", parent_version=ver1)
    ver2 = e2.split(":")[0].split()[0]
    rr = api.render(ver2, preset="reel_9x16")
    info = util.ffprobe(store.get_version(ver2)["output_path"])
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (vs["width"], vs["height"]) == (1080, 1920), rr

    d = api.diff(ver1, ver2)
    assert "xfade" in d and "-0.6s" in d.splitlines()[0], d
