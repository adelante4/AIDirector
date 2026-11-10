from __future__ import annotations
from pydantic import BaseModel


class Config(BaseModel):
    mode: str = "draft"            # draft | final
    max_cost_usd: float = 5.0       # hard cap per run
    n_shots: int = 6
    n_candidates: int = 1           # best-of-N keyframes (ViMax best-image-selector idea)
    max_retries: int = 1            # critic-triggered regenerations per shot
    pass_threshold: float = 0.6
    width: int = 640
    height: int = 360
    fps: int = 25
    seed: int = 7

    @classmethod
    def for_mode(cls, mode: str, **kw) -> "Config":
        if mode == "final":
            base = dict(mode="final", max_cost_usd=30.0, n_candidates=3, max_retries=2,
                        width=1280, height=720, pass_threshold=0.75)
        else:
            base = dict(mode="draft")
        base.update(kw)
        return cls(**base)
