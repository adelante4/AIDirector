from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Tracks spend per stage and enforces a hard cap."""

    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.by_stage: dict[str, float] = defaultdict(float)
        self.events: list[dict] = []

    @property
    def total(self) -> float:
        return round(sum(self.by_stage.values()), 6)

    def charge(self, stage: str, usd: float, note: str = "") -> None:
        if self.total + usd > self.cap:
            raise BudgetExceeded(f"{stage}: {self.total + usd:.3f} > cap {self.cap:.2f} USD")
        self.by_stage[stage] += usd
        self.events.append({"stage": stage, "usd": usd, "note": note})

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({"total": self.total, "by_stage": self.by_stage,
                                    "events": self.events}, indent=2))
