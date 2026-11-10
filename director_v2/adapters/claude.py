"""Claude adapters: structured output via forced tool call; vision critic."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import List, Type

from ..contracts import Bible, CriticScore, Shot, StyleCard
from ..prompts import CRITIC, SYSTEM, TASKS

# USD per million tokens (input, output). Rough estimates, adjust when pricing changes.
PRICE = {"in": 3.0, "out": 15.0}


def _client():
    import anthropic
    return anthropic.Anthropic()


def _cost(usage) -> float:
    return (usage.input_tokens * PRICE["in"] + usage.output_tokens * PRICE["out"]) / 1e6


def _tool(schema: Type):
    return {"name": "answer", "description": "Return the structured answer.", "input_schema": schema.model_json_schema()}


class ClaudeLLM:
    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.client = client or _client()
        self.ledger = None

    def structured(self, schema: Type, task: str, **ctx):
        payload = json.dumps(ctx, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o), indent=1)
        msg = self.client.messages.create(
            model=self.model, max_tokens=4096, system=SYSTEM,
            tools=[_tool(schema)], tool_choice={"type": "tool", "name": "answer"},
            messages=[{"role": "user", "content": f"{TASKS[task]}\n\nINPUT:\n{payload}"}])
        if self.ledger:
            self.ledger.charge("llm", _cost(msg.usage), task)
        block = next(b for b in msg.content if b.type == "tool_use")
        return schema.model_validate(block.input)


class ClaudeCritic:
    cost_per_call = 0.0   # charged from real token usage via the ledger

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.client = client or _client()
        self.ledger = None

    @staticmethod
    def _img(p: Path) -> dict:
        data = base64.standard_b64encode(Path(p).read_bytes()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    def score(self, shot: Shot, frames: List[Path], bible: Bible, style: StyleCard, attempt: int) -> CriticScore:
        names = [shot.location, *shot.characters, *shot.props]
        refs = [a for n in names if (a := bible.get(n)) and a.image_path]
        content: list = [{"type": "text", "text": "REFERENCE IMAGES (bible): " + ", ".join(a.name for a in refs)}]
        content += [self._img(Path(a.image_path)) for a in refs]
        content += [{"type": "text", "text": "GENERATED FRAMES (first, then last if present):"}]
        content += [self._img(f) for f in frames]
        content += [{"type": "text", "text": f"SHOT: {shot.model_dump_json()}\nSTYLE: {style.model_dump_json()}"}]
        msg = self.client.messages.create(
            model=self.model, max_tokens=512, system=CRITIC,
            tools=[_tool(CriticScore)], tool_choice={"type": "tool", "name": "answer"},
            messages=[{"role": "user", "content": content}])
        if self.ledger:
            self.ledger.charge("critic", _cost(msg.usage), f"shot{shot.idx}")
        block = next(b for b in msg.content if b.type == "tool_use")
        return CriticScore.model_validate(block.input)
