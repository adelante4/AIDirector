from typing import List

from pydantic import BaseModel, Field


class StoryOutput(BaseModel):
    story: str = Field(..., title="The generated story based on the post")


class StoryBoard(BaseModel):
    visual_theme: str = Field(..., title="The visual theme for the given story")
    scenes: List[str] = Field(..., title="The description of the scenes in the story")


class ScenePromptsOutput(BaseModel):
    prompts: List[str] = Field(..., title="The prompts for the given scene")

    def __str__(self):
        return "\n".join([f"prompt{i}: {prompt}" for i, prompt in enumerate(self.prompts)])
