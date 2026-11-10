from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

from .config import Config
from .contracts import (Asset, Bible, InfluencePick, PostInput, RenderResult, ShotList, ShotRender, Story,
                        StyleCard, Themes, validate_bible_covers_shots)
from .ledger import Ledger


def ingest(a, cfg: Config, led: Ledger, inp: PostInput) -> Themes:
    return a.llm.structured(Themes, "extract themes and tensions", post=inp.post, comments=inp.comments)


def write_story(a, cfg, led, inp: PostInput, themes: Themes) -> Story:
    # Keep the v1 rule: be inspired by the discussion, never retell the news.
    return a.llm.structured(Story, "abstract absurdist short story", post=inp.post, themes=themes)


def blend_style(a, cfg, led, story: Story) -> StyleCard:
    """Pick influences from the v1 catalog (90 books with briefs), then blend them into a style card.
    TODO: extend the catalog with films/painters and use embedding retrieval instead of LLM picking."""
    from src.director.constants import acclaimed_controversial_stories as catalog
    from src.director.constants import acclaimed_controversial_stories_briefs as briefs
    pick = a.llm.structured(InfluencePick, "pick two influences", story=story.text, catalog=sorted(set(catalog)))
    chosen = [t for t in pick.titles if t in briefs][:2] or sorted(briefs)[:2]
    notes = {t: briefs[t] for t in chosen}
    card = a.llm.structured(StyleCard, "style blend", story=story.text, influences=notes, weights=pick.weights)
    return card


def build_bible(a, cfg, led, story: Story, style: StyleCard, out: Path) -> Bible:
    """Lock characters/props/locations and render reference images ONCE, before any shot exists."""
    bible = a.llm.structured(Bible, "extract bible", story=story.text, style=style)
    for asset in bible.assets:
        img = a.image.generate(f"{asset.generation_prompt}. {bible.look}", [], out / f"{asset.kind}_{asset.name}.png",
                               cfg.seed, (cfg.width, cfg.height))
        led.charge("bible", a.image.cost_per_image, asset.name)
        asset.image_path = str(img)
    return bible


def plan_shots(a, cfg, led, story: Story, style: StyleCard, bible: Bible) -> ShotList:
    shots = a.llm.structured(ShotList, "shot list", story=story.text, style=style, bible=bible, n_shots=cfg.n_shots)
    validate_bible_covers_shots(bible, shots)   # structural rule, not a prompt suggestion
    return shots


def _refs_for(shot, bible: Bible, prev_last: Path | None) -> List[Path]:
    """ViMax-style reference selection: assets this shot needs + previous frame for continuity."""
    names = [shot.location, *shot.characters, *shot.props]
    refs = [Path(bible.get(n).image_path) for n in names if bible.get(n) and bible.get(n).image_path]
    if prev_last:
        refs.append(prev_last)
    return refs


def render(a, cfg: Config, led: Ledger, shots: ShotList, bible: Bible, style: StyleCard, out: Path) -> RenderResult:
    results: List[ShotRender] = []
    prev_last: Path | None = None
    size = (cfg.width, cfg.height)
    for shot in shots.shots:
        scores, attempt = [], 0
        while True:
            refs = _refs_for(shot, bible, prev_last)
            sdir = out / f"shot_{shot.idx:02d}"
            ff = a.image.generate(f"{shot.first_frame_desc}. {style.visual_theme}", refs,
                                  sdir / f"ff_a{attempt}.png", cfg.seed + attempt, size)
            led.charge("render", a.image.cost_per_image, f"shot{shot.idx} first frame")
            lf = None
            if shot.variation_type != "small":   # ViMax: only condition on a last frame when the shot changes a lot
                lf = a.image.generate(f"{shot.last_frame_desc}. {style.visual_theme}", refs,
                                      sdir / f"lf_a{attempt}.png", cfg.seed + attempt, size)
                led.charge("render", a.image.cost_per_image, f"shot{shot.idx} last frame")
            clip = a.video.generate(f"{shot.motion_desc}. {shot.visual_desc}", ff, lf, shot.duration_s,
                                    sdir / f"clip_a{attempt}.mp4", cfg.seed + attempt, cfg.fps)
            led.charge("render", a.video.cost_per_second * shot.duration_s, f"shot{shot.idx} clip")
            sc = a.critic.score(shot, [p for p in (ff, lf) if p], bible, style, attempt)
            led.charge("critic", a.critic.cost_per_call, f"shot{shot.idx}")
            scores.append(sc)
            if sc.total >= cfg.pass_threshold or attempt >= cfg.max_retries:
                break
            attempt += 1
        prev_last = lf or ff
        results.append(ShotRender(shot_idx=shot.idx, first_frame=str(ff), last_frame=str(lf) if lf else None,
                                  clip_path=str(clip), attempts=attempt + 1, scores=scores,
                                  passed=scores[-1].total >= cfg.pass_threshold))
    return RenderResult(shots=results)


def assemble(cfg: Config, result: RenderResult, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lst = out / "concat.txt"
    lst.write_text("".join(f"file '{Path(s.clip_path).resolve()}'\n" for s in result.shots))
    final = out / "final.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(final)], check=True)
    return final
