"""Export a version's timeline as OpenTimelineIO (.otio) so the cut opens in Resolve / Premiere."""
from __future__ import annotations

from pathlib import Path


def export(timeline: dict, src_path: str, out_path: Path, name: str) -> Path | None:
    try:
        import opentimelineio as otio
    except ImportError:
        return None
    from .render import frame_snap

    fps = float(timeline["fps"])
    segs = frame_snap(timeline["segments"], fps)
    trans = timeline.get("trans")
    ref = otio.schema.ExternalReference(target_url=Path(src_path).resolve().as_uri())
    tl = otio.schema.Timeline(name=name)
    tl.global_start_time = otio.opentime.RationalTime(0, fps)
    for kind in (otio.schema.TrackKind.Video, otio.schema.TrackKind.Audio):
        track = otio.schema.Track(name=kind, kind=kind)
        for i, (a, b) in enumerate(segs):
            if i and trans:
                half = otio.opentime.RationalTime(trans[1] / 2 * fps, fps)
                track.append(otio.schema.Transition(
                    transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve, in_offset=half, out_offset=half))
            # OTIO dissolves eat into neighbouring clips' handles instead of overlapping them, so
            # shorten each clip by d/2 at every transition to match the rendered duration
            if trans:
                a += trans[1] / 2 if i > 0 else 0
                b -= trans[1] / 2 if i < len(segs) - 1 else 0
            rng = otio.opentime.TimeRange(otio.opentime.RationalTime(round(a * fps), fps),
                                          otio.opentime.RationalTime(round((b - a) * fps), fps))
            track.append(otio.schema.Clip(name=f"seg{i + 1}", media_reference=ref.clone(), source_range=rng))
        tl.tracks.append(track)
    if abs((timeline.get("speed") or 1) - 1) > 1e-3:
        tl.metadata["vidx"] = {"speed": timeline["speed"], "note": "global speed not encoded as clip effects"}
    if timeline.get("crop"):
        tl.metadata.setdefault("vidx", {})["crop"] = timeline["crop"]
    out_path = out_path.with_suffix(".otio")
    otio.adapters.write_to_file(tl, str(out_path))
    return out_path
