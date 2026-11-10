"""Typed contracts between pipeline stages.

Ideas borrowed:
- VideoMemory: per-asset memory bank (generation prompt + image path + refs).
- ArcReel: the bible (characters/props/locations) is LOCKED before any shot is planned.
- ViMax: first/last frame per shot, variation_type, <Name> tags in visual_desc.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class PostInput(BaseModel):
    post: str
    comments: List[str] = []


class Themes(BaseModel):
    themes: List[str]
    tensions: List[str]
    mood: str


class Story(BaseModel):
    title: str
    text: str


class StyleCard(BaseModel):
    """Output of the style-blend stage. Conditions every downstream stage."""
    influences: List[str]
    weights: List[float] = []
    palette: str
    camera_language: str
    pacing: str
    tone: str
    visual_theme: str


class Asset(BaseModel):
    kind: Literal["character", "prop", "location"]
    name: str
    description: str
    generation_prompt: str
    image_path: Optional[str] = None


class Bible(BaseModel):
    look: str
    assets: List[Asset]

    def get(self, name: str) -> Optional[Asset]:
        return next((a for a in self.assets if a.name == name), None)


class Shot(BaseModel):
    idx: int
    act: int = 1
    narrative_function: str
    location: str
    characters: List[str] = []
    props: List[str] = []
    shot_size: str = "medium"
    camera_move: str = "static"
    visual_desc: str = Field(description="Use <Name> tags for bible assets")
    first_frame_desc: str
    last_frame_desc: str
    variation_type: Literal["large", "medium", "small"] = "small"
    motion_desc: str
    audio_desc: Optional[str] = None
    duration_s: float = 4.0


class ShotList(BaseModel):
    shots: List[Shot]

    @model_validator(mode="after")
    def _idx_sequential(self):
        if [s.idx for s in self.shots] != list(range(len(self.shots))):
            raise ValueError("shot idx must be 0..n-1 in order")
        return self


class CriticScore(BaseModel):
    prompt_match: float
    consistency: float
    style: float
    notes: str = ""

    @property
    def total(self) -> float:
        return (self.prompt_match + self.consistency + self.style) / 3


class ShotRender(BaseModel):
    shot_idx: int
    first_frame: str
    last_frame: Optional[str] = None
    clip_path: str
    attempts: int
    scores: List[CriticScore] = []
    passed: bool = True


class RenderResult(BaseModel):
    shots: List[ShotRender]


class RunReport(BaseModel):
    run_id: str
    mode: str
    total_cost_usd: float
    cost_by_stage: dict
    retries: int
    final_video: str
    avg_critic_score: float


def validate_bible_covers_shots(bible: Bible, shots: ShotList) -> None:
    """Structural rule: a shot may only reference assets that exist in the bible."""
    known = {a.name for a in bible.assets}
    for s in shots.shots:
        missing = ({s.location} | set(s.characters) | set(s.props)) - known
        if missing:
            raise ValueError(f"shot {s.idx} references assets not in bible: {sorted(missing)}")
