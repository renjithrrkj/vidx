"""Frame grabs and labeled contact sheets."""
from __future__ import annotations

import io
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from .util import mmss


def grab_frame(path: str, t: float, long_side: int = 384) -> Image.Image | None:
    """Single frame at time t (fast input seek), longest side scaled to long_side."""
    vf = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", path,
           "-frames:v", "1", "-vf", vf, "-f", "image2pipe", "-c:v", "png", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        if t > 0.5:  # past the last decodable frame: back off a little
            return grab_frame(path, t - 0.5, long_side)
        return None
    return Image.open(io.BytesIO(p.stdout)).convert("RGB")


def grab_frames(path: str, times: list[float], long_side: int = 384, workers: int = 4) -> list[Image.Image | None]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: grab_frame(path, t, long_side), times))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(frames: list[Image.Image | None], labels: list[str], cols: int | None = None) -> Image.Image:
    """Grid of frames, each with its label burned into the top-left corner."""
    n = len(frames)
    cols = cols or min(4, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    ok = [f for f in frames if f is not None]
    tw, th = (max(f.width for f in ok), max(f.height for f in ok)) if ok else (384, 216)
    gap = 4
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(14, th // 14))
    for i, (f, lab) in enumerate(zip(frames, labels)):
        x, y = (i % cols) * (tw + gap), (i // cols) * (th + gap)
        if f is not None:
            sheet.paste(f, (x + (tw - f.width) // 2, y + (th - f.height) // 2))
        else:
            draw.text((x + 10, y + th // 2), "no frame", fill=(200, 80, 80), font=font)
        bbox = draw.textbbox((x + 6, y + 4), lab, font=font)
        draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0))
        draw.text((x + 6, y + 4), lab, fill=(255, 230, 0), font=font)
    return sheet


def label(t: float, shot: str | None = None) -> str:
    return f"{shot} {mmss(t)}" if shot else mmss(t)
