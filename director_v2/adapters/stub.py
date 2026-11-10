"""Offline stubs: no keys, no cost. Produce real files so the whole pipeline (incl. ffmpeg) runs."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
from typing import List, Optional, Type

from PIL import Image, ImageDraw

from ..contracts import (Asset, Bible, CriticScore, Shot, ShotList, Story, StyleCard, Themes)


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


class StubLLM:
    def structured(self, schema: Type, task: str, **ctx):
        post = ctx.get("post", "")
        key = (post or ctx.get("story", ""))[:40]
        if schema is Themes:
            return Themes(themes=["disconnection", "collective noise"], tensions=["speak vs. be heard"],
                          mood="dry, uneasy")
        if schema is Story:
            return Story(title=f"The Committee of Echoes ({key[:20]})",
                         text="Mara and the Clerk argue in a corridor that rearranges itself each time they agree.")
        if schema is StyleCard:
            return StyleCard(influences=["Kafka", "Lynch"], weights=[0.6, 0.4],
                             palette="sodium yellow, bruised green", camera_language="slow dolly, wide lenses",
                             pacing="slow, sudden cuts", tone="deadpan absurd",
                             visual_theme="bureaucratic dream corridors")
        if schema is Bible:
            return Bible(look="grainy 35mm, sodium light", assets=[
                Asset(kind="character", name="Mara", description="tired woman in a green coat",
                      generation_prompt="portrait, tired woman, green coat, sodium light"),
                Asset(kind="character", name="Clerk", description="faceless clerk in grey suit",
                      generation_prompt="portrait, faceless clerk, grey suit"),
                Asset(kind="prop", name="Stamp", description="oversized rubber stamp",
                      generation_prompt="oversized rubber stamp on desk"),
                Asset(kind="location", name="Corridor", description="endless office corridor",
                      generation_prompt="endless office corridor, sodium light, wide lens"),
            ])
        if schema is ShotList:
            n = ctx.get("n_shots", 6)
            shots = []
            for i in range(n):
                shots.append(Shot(
                    idx=i, act=1 + i * 3 // n, narrative_function=["setup", "conflict", "twist"][i * 3 // n],
                    location="Corridor", characters=["Mara"] + (["Clerk"] if i % 2 else []),
                    props=["Stamp"] if i % 3 == 2 else [], shot_size=["wide", "medium", "close"][i % 3],
                    camera_move=["dolly in", "static", "pan"][i % 3],
                    visual_desc=f"<Mara> in the corridor, beat {i}",
                    first_frame_desc=f"<Mara> stands in the corridor, beat {i} start",
                    last_frame_desc=f"<Mara> turns, beat {i} end",
                    variation_type="medium" if i % 3 == 1 else "small",
                    motion_desc="slow push in", duration_s=3.0))
            return ShotList(shots=shots)
        raise NotImplementedError(schema)


class StubImageGen:
    cost_per_image = 0.0

    def generate(self, prompt: str, refs: List[Path], out: Path, seed: int, size) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        r, g, b = (_h(prompt + str(seed)) >> k & 0xFF for k in (0, 8, 16))
        img = Image.new("RGB", size, (r // 2 + 40, g // 2 + 40, b // 2 + 40))
        d = ImageDraw.Draw(img)
        d.text((12, 12), prompt[:70], fill=(255, 255, 255))
        d.text((12, size[1] - 24), f"refs: {len(refs)}", fill=(255, 255, 255))
        img.save(out)
        return out


class StubVideoGen:
    cost_per_second = 0.0

    def generate(self, prompt: str, first_frame: Path, last_frame: Optional[Path],
                 seconds: float, out: Path, seed: int, fps: int) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = max(1, int(seconds * fps))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(first_frame),
               "-vf", f"zoompan=z='min(zoom+0.0015,1.3)':d={frames}:s=640x360:fps={fps},format=yuv420p",
               "-t", str(seconds), "-c:v", "libx264", str(out)]
        subprocess.run(cmd, check=True)
        return out


class StubCritic:
    cost_per_call = 0.0

    def score(self, shot: Shot, frames: List[Path], bible: Bible, style: StyleCard, attempt: int) -> CriticScore:
        # Shot 1 fails its first attempt on purpose so the retry loop is exercised.
        bad = shot.idx == 1 and attempt == 0
        v = 0.4 if bad else 0.85
        return CriticScore(prompt_match=v, consistency=v, style=v,
                           notes="stub: forced failure" if bad else "stub: ok")
