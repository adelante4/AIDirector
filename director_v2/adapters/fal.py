"""fal.ai adapters. Model IDs and prices are env-configurable because they change often."""
from __future__ import annotations
import os
import urllib.request
from pathlib import Path
from typing import List, Optional


def _fal():
    import fal_client
    return fal_client


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    return out


class FalImageGen:
    """Nano Banana: text-to-image without refs, edit endpoint with refs (identity/style carry-over)."""

    def __init__(self, t2i: str | None = None, edit: str | None = None, cost_per_image: float | None = None, fal=None):
        self.t2i = t2i or os.environ.get("FAL_IMAGE_T2I", "fal-ai/nano-banana")
        self.edit = edit or os.environ.get("FAL_IMAGE_EDIT", "fal-ai/nano-banana/edit")
        self.cost_per_image = cost_per_image if cost_per_image is not None else float(os.environ.get("FAL_IMAGE_COST", "0.04"))
        self.fal = fal or _fal()

    def generate(self, prompt: str, refs: List[Path], out: Path, seed: int, size) -> Path:
        if refs:
            urls = [self.fal.upload_file(str(r)) for r in refs[:6]]
            res = self.fal.subscribe(self.edit, arguments={
                "prompt": f"Keep the identity and look of the referenced subjects. {prompt}",
                "image_urls": urls, "num_images": 1, "output_format": "png"})
        else:
            res = self.fal.subscribe(self.t2i, arguments={"prompt": prompt, "num_images": 1, "output_format": "png"})
        return _download(res["images"][0]["url"], out)


class FalVideoGen:
    """Image-to-video. Default Wan 3.0 (start_image_url / end_image_url). Override FAL_VIDEO_MODEL to A/B models."""

    def __init__(self, model: str | None = None, cost_per_second: float | None = None, resolution: str | None = None, fal=None):
        self.model = model or os.environ.get("FAL_VIDEO_MODEL", "alibaba/wan-3.0/image-to-video")
        self.cost_per_second = cost_per_second if cost_per_second is not None else float(os.environ.get("FAL_VIDEO_COST_PER_S", "0.10"))
        self.resolution = resolution or os.environ.get("FAL_VIDEO_RES", "720p")
        self.fal = fal or _fal()

    def generate(self, prompt: str, first_frame: Path, last_frame: Optional[Path], seconds: float,
                 out: Path, seed: int, fps: int) -> Path:
        args = {"prompt": prompt, "start_image_url": self.fal.upload_file(str(first_frame)),
                "resolution": self.resolution, "duration": int(round(seconds)), "aspect_ratio": "16:9", "seed": seed}
        if last_frame:
            args["end_image_url"] = self.fal.upload_file(str(last_frame))
        res = self.fal.subscribe(self.model, arguments=args)
        return _download(res["video"]["url"], out)
