import logging

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram

from src.data.models.api_models import RequestAPIObject
from src.data.models.response_models import StoryOutput, StoryBoard, ScenePromptsOutput, RewrittenStory, \
    EnhancedPromptOutput
from src.utils.fixing_output_parser import FixingOutputParser
from src.director.prompts import story_board_template, story_template, scene_prompts_template, rewrite_story_template, \
    prompt_enhance_template


class Director:
    def __init__(self, request_data: RequestAPIObject,
                 story_retriever):
        self.request_data = request_data
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Director instance created")
        self.story_retriever = story_retriever

    def build_animation(self):
        self.logger.info("Starting animation building...")

        story = self.generate_story()
        self.request_data.progress_bar.progress(0.2)
        rewritten_story, similar_stories, artistic_style = self.rewrite_story(story, self.story_retriever)
        self.request_data.progress_bar.progress(0.4)
        story_board = self.generate_story_board(rewritten_story, artistic_style)
        self.request_data.progress_bar.progress(0.6)
        scene_prompts = self.generate_scene_prompts(story_board)
        self.request_data.progress_bar.progress(0.8)
        self.enhance_prompts(scene_prompts, story_board.visual_theme)
        self.request_data.progress_bar.progress(1)

        self.logger.info("Animation building complete")
        return scene_prompts, story, story_board, rewritten_story, similar_stories, artistic_style

    @staticmethod
    def get_completion_program(prompt_template_str, output_cls, output_parser=None):
        if output_parser:
            return LLMTextCompletionProgram.from_defaults(
                output_cls=output_cls,
                prompt_template_str=prompt_template_str,
                output_parser=output_parser,
                verbose=True,
            )
        else:
            return LLMTextCompletionProgram.from_defaults(
                output_cls=output_cls,
                prompt_template_str=prompt_template_str,
                verbose=True,
            )

    def generate_story(self) -> str:
        self.logger.info("Generating story...")
        story_program = self.get_completion_program(
            output_parser=FixingOutputParser(StoryOutput),
            prompt_template_str=story_template.get_template(),
            output_cls=StoryOutput
        )

        comments_str = "\n".join(f"<Comment>: {comment}" for comment in self.request_data.comments)
        story = story_program(post=self.request_data.post,
                              comments=comments_str,
                              format_instructions=PydanticOutputParser(StoryOutput).get_format_string())
        return story.story

    def generate_story_board(self, story: str, artistic_style: str = None):
        self.logger.info("Generating story board...")
        story_board_program = self.get_completion_program(
            output_parser=FixingOutputParser(StoryBoard),
            prompt_template_str=story_board_template.get_template(),
            output_cls=StoryBoard
        )

        story_board = story_board_program(story=story,
                                          artistic_style=artistic_style,
                                          format_instructions=PydanticOutputParser(StoryBoard).get_format_string())

        return story_board

    def generate_scene_prompts(self, story_board):
        self.logger.info("Generating scene prompts...")
        scene_prompts = {}
        for idx, scene in enumerate(story_board.scenes):
            scene_prompt = self.get_completion_program(
                output_parser=FixingOutputParser(ScenePromptsOutput),
                prompt_template_str=scene_prompts_template.get_template(),
                output_cls=ScenePromptsOutput
            )

            prompts = scene_prompt(visual_theme=story_board.visual_theme,
                                   scene_description=scene,
                                   format_instructions=PydanticOutputParser(ScenePromptsOutput).get_format_string())

            scene_prompts[f"scene_{idx}"] = prompts.prompts

        return scene_prompts

    @staticmethod
    def rewrite_story(story, story_retriever: VectorIndexRetriever):
        story_retriever.similarity_top_k = 50
        similar_stories = story_retriever.retrieve(story)[:2]

        if len(similar_stories) != 2:
            logging.info("Not enough similar stories found, could not rewrite the story")
            return story, []

        similar_stories_str = ""
        for idx, similar_story in enumerate(similar_stories):
            similar_stories_str += f"{idx}. **{similar_story.node.metadata['story']}**:\n"
            for k, v in similar_story.metadata.items():
                similar_stories_str += f"{k}: {v}\n"

        rewrite_program = Director.get_completion_program(
            output_parser=FixingOutputParser(RewrittenStory),
            prompt_template_str=rewrite_story_template.get_template(),
            output_cls=RewrittenStory,
        )
        rewritten_story = rewrite_program(movies=similar_stories_str,
                                          story_idea=story,
                                          format_instructions=PydanticOutputParser(RewrittenStory).get_format_string())

        return (rewritten_story.story_title + "\n" + rewritten_story.story_content, similar_stories,
                rewritten_story.artistic_style)

    def enhance_prompts(self, prompts, visual_theme):
        self.logger.info("Enhancing prompts...")

        enhance_program = Director.get_completion_program(
            output_parser=FixingOutputParser(EnhancedPromptOutput),
            prompt_template_str=prompt_enhance_template.get_template(),
            output_cls=EnhancedPromptOutput,
        )

        for scene, scene_prompts in prompts.items():
            for idx, prompt in enumerate(scene_prompts):
                enhanced_prompt = enhance_program(original_prompt=prompt,
                                                  visual_theme=visual_theme,
                                                  format_instructions=PydanticOutputParser(
                                                      EnhancedPromptOutput).get_format_string())
                prompts[scene][idx] = enhanced_prompt.enhanced_prompt

        return prompts
