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


class RewrittenStory(BaseModel):
    story_title: str = Field(..., title="Generated story title")
    story_content: str = Field(..., title="Generated story content")
    artistic_style: str = Field(..., title="Artistic style used for the rewritten story")


class StoryBriefOutput(BaseModel):
    brief: str = Field(..., title="Brief of the story")
    story_analysis: str = Field(..., title="Sociological and philosophical analysis of the characters in the story")
    artistic_style: str = Field(..., title="Artistic style of the art piece")


class EnhancedPromptOutput(BaseModel):
    enhanced_prompt: str = Field(..., title="Enhanced prompt for the given scene")
