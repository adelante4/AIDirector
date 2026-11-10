import argparse
import json
from pathlib import Path

from .adapters.base import Adapters
from .adapters.stub import StubCritic, StubImageGen, StubLLM, StubVideoGen
from .config import Config
from .contracts import PostInput
from .pipeline import run


def main():
    p = argparse.ArgumentParser(description="AIDirector v2: post in, video out")
    p.add_argument("--post", required=True, help="text file: first line = post, remaining lines = comments")
    p.add_argument("--mode", default="draft", choices=["draft", "final"])
    p.add_argument("--run-id", default="run")
    p.add_argument("--max-cost", type=float)
    p.add_argument("--fresh", action="store_true", help="ignore checkpoints")
    args = p.parse_args()

    lines = [l for l in Path(args.post).read_text().splitlines() if l.strip()]
    inp = PostInput(post=lines[0], comments=lines[1:])
    kw = {"max_cost_usd": args.max_cost} if args.max_cost else {}
    cfg = Config.for_mode(args.mode, **kw)
    adapters = Adapters(llm=StubLLM(), image=StubImageGen(), video=StubVideoGen(), critic=StubCritic())
    report = run(inp, adapters, cfg, run_id=args.run_id, resume=not args.fresh)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
