"""Provider interfaces. Swap implementations by config; the pipeline never imports a vendor."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

from ..contracts import Bible, CriticScore, Shot, StyleCard

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def structured(self, schema: Type[T], task: str, **ctx) -> T: ...


class ImageGen(Protocol):
    cost_per_image: float
    def generate(self, prompt: str, refs: List[Path], out: Path, seed: int,
                 size: tuple[int, int]) -> Path: ...


class VideoGen(Protocol):
    cost_per_second: float
    def generate(self, prompt: str, first_frame: Path, last_frame: Optional[Path],
                 seconds: float, out: Path, seed: int, fps: int) -> Path: ...


class Critic(Protocol):
    cost_per_call: float
    def score(self, shot: Shot, frames: List[Path], bible: Bible, style: StyleCard,
              attempt: int) -> CriticScore: ...


class Adapters(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: object
    image: object
    video: object
    critic: object
