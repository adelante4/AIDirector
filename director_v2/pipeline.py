"""Checkpointed runner. Each stage writes output/<run_id>/NN_name.json; reruns resume from disk.
Swap this for a LangGraph graph once the stages settle; the contracts stay the same."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Type, TypeVar

from pydantic import BaseModel

from .adapters.base import Adapters
from .config import Config
from .contracts import (Bible, PostInput, RenderResult, RunReport, ShotList, Story, StyleCard, Themes)
from . import stages
from .ledger import Ledger

T = TypeVar("T", bound=BaseModel)


def _cached(path: Path, model: Type[T], fn: Callable[[], T], resume: bool) -> T:
    if resume and path.exists():
        return model.model_validate_json(path.read_text())
    obj = fn()
    path.write_text(obj.model_dump_json(indent=2))
    return obj


def run(inp: PostInput, adapters: Adapters, cfg: Config, run_id: str = "run",
        root: Path = Path("output"), resume: bool = True) -> RunReport:
    out = root / run_id
    out.mkdir(parents=True, exist_ok=True)
    led = Ledger(cfg.max_cost_usd)
    a = adapters
    if hasattr(a.llm, "ledger"):
        a.llm.ledger = led
    if hasattr(a.critic, "ledger"):
        a.critic.ledger = led

    themes = _cached(out / "01_themes.json", Themes, lambda: stages.ingest(a, cfg, led, inp), resume)
    story = _cached(out / "02_story.json", Story, lambda: stages.write_story(a, cfg, led, inp, themes), resume)
    style = _cached(out / "03_style.json", StyleCard, lambda: stages.blend_style(a, cfg, led, story), resume)
    bible = _cached(out / "04_bible.json", Bible, lambda: stages.build_bible(a, cfg, led, story, style, out / "bible"), resume)
    shots = _cached(out / "05_shots.json", ShotList, lambda: stages.plan_shots(a, cfg, led, story, style, bible), resume)
    result = _cached(out / "06_render.json", RenderResult,
                     lambda: stages.render(a, cfg, led, shots, bible, style, out / "render"), resume)
    final = stages.assemble(cfg, result, out)

    scores = [s.scores[-1].total for s in result.shots]
    report = RunReport(run_id=run_id, mode=cfg.mode, total_cost_usd=led.total, cost_by_stage=dict(led.by_stage),
                       retries=sum(s.attempts - 1 for s in result.shots), final_video=str(final),
                       avg_critic_score=round(sum(scores) / len(scores), 3))
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    led.dump(out / "ledger.json")
    return report
