import logging

from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram

from src.data.models.api_models import RequestAPIObject
from src.data.models.response_models import StoryOutput, StoryBoard, ScenePromptsOutput
from src.director.prompts import story_board_template, story_template, scene_prompts_template


class Director:
    def __init__(self, request_data: RequestAPIObject):
        self.request_data = request_data
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Director instance created")

    def build_animation(self):
        self.logger.info("Starting animation building...")

        story = self.generate_story()
        self.request_data.progress_bar.progress(0.3)
        story_board = self.generate_story_board(story)
        self.request_data.progress_bar.progress(0.6)
        scene_prompts = self.generate_scene_prompts(story_board)
        self.request_data.progress_bar.progress(1)

        self.logger.info("Animation building complete")
        return scene_prompts, story, story_board

    @staticmethod
    def get_completion_program(prompt_template_str, output_cls):
        return LLMTextCompletionProgram.from_defaults(
            output_cls=output_cls,
            prompt_template_str=prompt_template_str,
            verbose=True,
        )

    def generate_story(self) -> str:
        self.logger.info("Generating story...")
        story_program = self.get_completion_program(
            prompt_template_str=story_template.get_template(),
            output_cls=StoryOutput
        )

        comments_str = "\n".join(f"<Comment>: {comment}" for comment in self.request_data.comments)
        story = story_program(post=self.request_data.post,
                              comments=comments_str)
        return story.story

    def generate_story_board(self, story: str):
        self.logger.info("Generating story board...")
        story_board_program = self.get_completion_program(
            prompt_template_str=story_board_template.get_template(),
            output_cls=StoryBoard
        )

        story_board = story_board_program(story=story)

        return story_board

    def generate_scene_prompts(self, story_board):
        self.logger.info("Generating scene prompts...")
        scene_prompts = {}
        for idx, scene in enumerate(story_board.scenes):
            scene_prompt = self.get_completion_program(
                prompt_template_str=scene_prompts_template.get_template(),
                output_cls=ScenePromptsOutput
            )

            prompts = scene_prompt(visual_theme=story_board.visual_theme,
                                   scene_description=scene)

            scene_prompts[f"scene_{idx}"] = str(prompts)

        return scene_prompts
