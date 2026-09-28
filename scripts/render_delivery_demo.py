"""Render actual recorded terminal evidence to a 90-second captioned MP4.

Presentation-only dependencies: Pillow==11.3.0, imageio-ffmpeg==0.6.0.
Not a realtime speed claim: pauses/output are shortened, all source stdout and
exit codes stay available in the accompanying terminal.json.
"""
import argparse
import json
from pathlib import Path
import textwrap

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("transcript", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--font", required=True, type=Path)
    a = p.parse_args()
    doc = json.loads(a.transcript.read_text())
    events = doc["events"]
    if doc["kind"] != "actual_command_recording" or len(events) != 4:
        raise ValueError("expected the four actual recorded stages")
    if any(e.get("returncode") != 0 for e in events):
        raise ValueError("cannot render a successful demo from failed commands")
    if "VALID: all output files satisfy contract C6" not in "".join(c["text"] for c in events[-1]["chunks"]):
        raise ValueError("actual validator PASS is missing")
    if a.output.exists():
        raise FileExistsError(a.output)
    w, h, fps = 1600, 1000, 10
    font = lambda size: ImageFont.truetype(str(a.font), size)
    title, body, small = font(31), font(22), font(18)
    durations = [16, 32, 20, 22]
    writer = imageio_ffmpeg.write_frames(str(a.output), (w, h), fps=fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart", "-crf", "20"], macro_block_size=2)
    writer.send(None)
    try:
        for index, (event, duration) in enumerate(zip(events, durations)):
            im = Image.new("RGB", (w, h), "#101622")
            d = ImageDraw.Draw(im)
            d.text((55, 36), "DEEPTECH AI  /  VEHICLE RE-ID", font=title, fill="#edf3ff")
            d.text((55, 88), "DINO392 + CLIP288 · verifier392 · D3072 · w=0.15 · threshold=6.6", font=body, fill="#98baff")
            d.rounded_rectangle((40, 142, w - 40, 890), radius=18, fill="#080e18", outline="#344761", width=2)
            d.text((65, 164), f"0{index + 1} / 04   {event['title']}", font=title, fill="#72e1c1")
            command = event["command"]
            # Literal recorded command, wrapped for display; no invented prompt activity.
            lines = textwrap.wrap("$ " + command, width=112, break_long_words=True)
            y = 226
            for line in lines:
                d.text((65, y), line, font=body, fill="#d8e5fb")
                y += 30
            d.line((65, y + 9, w - 65, y + 9), fill="#344761", width=1)
            y += 35
            output = "".join(c["text"] for c in event["chunks"]).strip().splitlines()
            wrapped = [part for line in output for part in (textwrap.wrap(line, 112) or [""])]
            capacity = max(1, (792 - y) // 28)
            if len(wrapped) > capacity:
                wrapped = ["[... полный вывод сохранён в terminal.json ...]"] + wrapped[-(capacity - 1):]
            for line in wrapped:
                d.text((65, y), line, font=body, fill="#aee9d0" if "VALID:" in line else "#c5d2e4")
                y += 28
            elapsed = event["ended_seconds"] - event["started_seconds"]
            d.text((65, 841), f"exit code: {event['returncode']}  |  фактическое время команды: {elapsed:.2f} с", font=body, fill="#72e1c1")
            d.text((55, 917), f"Реальный Docker-прогон · train: {doc['demo']['query']} query + {doc['demo']['gallery']} gallery · сеть отключена", font=small, fill="#b6c7df")
            d.text((55, 947), "Паузы и длинный вывод сокращены для показа. Это не замер скорости и не оценка качества.", font=small, fill="#879bb8")
            frame = im.tobytes()
            for _ in range(duration * fps):
                writer.send(frame)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
