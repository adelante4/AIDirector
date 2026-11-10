import json
import shutil
import subprocess
from pathlib import Path

import pytest

from director_v2.adapters.base import Adapters
from director_v2.adapters.stub import StubCritic, StubImageGen, StubLLM, StubVideoGen
from director_v2.config import Config
from director_v2.contracts import Bible, PostInput, Shot, ShotList, validate_bible_covers_shots
from director_v2.ledger import BudgetExceeded, Ledger
from director_v2.pipeline import run


def _adapters():
    return Adapters(llm=StubLLM(), image=StubImageGen(), video=StubVideoGen(), critic=StubCritic())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_end_to_end_produces_playable_mp4(tmp_path):
    rep = run(PostInput(post="City bans noise", comments=["lol", "ok"]), _adapters(), Config(), "t", tmp_path)
    mp4 = Path(rep.final_video)
    assert mp4.exists() and mp4.stat().st_size > 0
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", str(mp4)]))
    assert dur > 10
    assert rep.retries == 1          # stub critic fails shot 1 once


def test_resume_skips_completed_stages(tmp_path):
    inp = PostInput(post="x")
    run(inp, _adapters(), Config(), "r", tmp_path)
    story_file = tmp_path / "r" / "02_story.json"
    story = json.loads(story_file.read_text())
    story["title"] = "EDITED"
    story_file.write_text(json.dumps(story))
    run(inp, _adapters(), Config(), "r", tmp_path)
    assert json.loads(story_file.read_text())["title"] == "EDITED"


def test_budget_cap():
    led = Ledger(1.0)
    led.charge("render", 0.9)
    with pytest.raises(BudgetExceeded):
        led.charge("render", 0.2)


def test_shots_must_reference_bible_assets():
    bible = Bible(look="x", assets=[])
    sl = ShotList(shots=[Shot(idx=0, narrative_function="a", location="Nowhere", visual_desc="v",
                              first_frame_desc="f", last_frame_desc="l", motion_desc="m")])
    with pytest.raises(ValueError, match="not in bible"):
        validate_bible_covers_shots(bible, sl)
