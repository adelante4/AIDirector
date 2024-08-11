from llama_index.core import PromptTemplate

story_template = (
    """
    Imagine you're a very dark, experienced writer with a very absurd view of life. You are an avant-garde artist
    whose works are always different from his generation. The genre of your works can be described as 
    post-modern inspired by the living situations.
 
    For your new story, You'll be given one social media post with some people's comments. 
    Your task is to extract an innovative short story with the main ideas as the whole discussion.
    
    **Instructions**
    1. Try to reflect on the interaction between such characters in your story.
    2. The short story should be around 2-3 paragraphs long.
    3. Get creative with it. Remember what kind of writer you are.
    4. Use fictional characters instead of each group.
    5. The subject of your story should not be very similar to the news, the similarity should be in the deeper, abstract levels of the story.
    6. The same should apply to the comments, you should not mention exactly the same stuff, you should just be inspired by them.
    7. You don't have to use all of the comments.
    
    You'll  be given the information in the following format:
    <Post>: the header
    
    <Comment>: some comment from a user
    <Comment>: some comment from another user
    ...
    
    **Input**
    <Post>: {post}
    
    {comments}
    
    ONLY provide the json, no extra words around it.
    Don't use characters which can be interpreted as control characters in json format. like \n, \t, \r etc.
    {format_instructions}
    """
)

story_board_template = (
    """
    Imagine you're a very artistic short-animation director, people define your artistic style as:
    {artistic_style} 
    You'll be provided with a short story, your task is to make a storyboard for this story. 
    
    **Instructions**
    1. Divide the story into as many cinematic scenes as needed. 
    2. Describe each scene in a way that it's straightforward enough for your illustrator to draw them. 
    3. Add a short note for the art director to describe the whole visual theme of the animation.
    
    **Input Story**
    {story}
    
    ONLY provide the json, no extra words around it.
    Don't use characters which can be interpreted as control characters in json format. like \n, \t, \r etc.
    {format_instructions}
    """
)

scene_prompts_template = (
    """
    You're an experienced, creative visual designer working with stable diffusion models as the tool to build your projects. In this project, you create an animated movie scene from a scene description. You build multiple AI-generated pictures and then interpolate between them to create the motion.
    You'll be provided with a visual theme and a scene description from a storyboard. Your task is to provide prompts for each of the pictures to be generated.
    
    **Instructions**:
    1. First, you should extract the objects in the scene. 
    2. Write prompts to create pictures depicting the objects.
    3. Each prompt should be independent of the others and should contain all the information to create a picture.
    4. Do not reference to the object in the previous prompts generated. describe the objects in all the prompts
    5. The pictures should take the chronological order of events in the scene into account.
    6. keep in mind you should create multiple pictures that the interpolation between them, creates the given scene.
    7. Pay attention to the given visual theme and build your prompts in that style.
    
    You'll be given the information in the following format:
    <Visual Theme>:
    the visual theme describing the vibe and theme of the scene
    <Scene description>:
    The description of the scene in the storyboard, written by the director.
    
    
    **Input**:
    <Visual Theme>:
    {visual_theme}
        
    <Scene description>:
    {scene_description}
    
    ONLY provide the json, no extra words around it.
    Don't use characters which can be interpreted as control characters in json format. like \n, \t, \r etc.
    {format_instructions}
    """
)

story_brief_template = """
Imagine you are a professor of literature and you are teaching a class on postmodern literature.
Give a brief of {story}.
The brief should contain the main plot points of the story narrated in a concise manner.
provide an sociological and philosophical analysis about the characters in the story.
also describe the artistic style of this art piece.
Provide the answer in the given json format.
Don't use characters which can be interpreted as control characters in json format. like \n, \t, \r etc.
"""

rewrite_story_template = """
You will be provided with descriptions of two movies, including their story analyses and the artistic styles of their directors. 
Additionally, you will receive a simple story idea.

Your task is to rewrite this story idea into an original narrative that draws abstract inspiration from both movies and 
their directors' styles, while ensuring the final story stands as a unique creation.

**Follow these steps**:

    - Abstract Inspiration: Use the provided story idea only as a conceptual starting point. Your final story should be distinctly original, not a direct adaptation.

    - Movie Influence: Subtly incorporate thematic elements, visual motifs, or narrative techniques from the provided films without directly copying their plots. The influences should be organic and enhance your narrative.

    - Character Development: Create complex, multidimensional characters with authentic motivations and conflicts. These characters should drive the narrative and feel fully realized.

    - Atmosphere and Tone: Establish a distinct atmosphere and tone that reflects a hybrid of the artistic styles of both directors. This should permeate the entire story.

    - Vivid Imagery: Use vivid, sensory language to create a rich, immersive story world. This should evoke the visual and emotional impact of a well-directed film.

    - Narrative Structure: Structure your story with a balance of tension, pacing, and emotional resonance. Ensure the narrative unfolds in a way that keeps readers engaged.

    - Themes and Commentary: Explore thought-provoking themes or social commentary that is relevant to contemporary audiences. This should add depth to your story.

    - Genre Subversion: Subvert traditional genre expectations or blend multiple genres to create something innovative and fresh.

    - Authentic Dialogue: Write dialogue that feels natural and serves to reveal character depth and advance the plot.

    - Satisfying Resolution: Conclude the story with a resolution that is both satisfying and leaves room for interpretation, inviting readers to think beyond the final words.

**Output Requirements**:

Your story must be highly creative, demonstrating the potential to captivate discerning readers or critics.
While drawing inspiration from the provided films, ensure these influences are subtle and serve your unique narrative vision. Avoid forced connections or overt references.
Write in a style that is distinctly human, with nuanced observations, idiosyncratic details, and a clear authorial voice. Your prose should flow naturally, avoiding formulaic structures or patterns that might suggest computer-generated text.
Also provide the artistic style needed for the director that will direct this story.


**Input**:
**New Story Idea**: {story_idea}
**Movies**: 
{movies}


Provide the rewritten story in the given JSON format.
Only provide the JSON output, with no extra words or explanations around it.
{format_instructions}
"""

output_fixing_template = """
Fix the given json output to satisfy the constraints laid out in the Instructions.
Instructions:
--------------
{instructions}
--------------
json output:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions.
Don't use characters which can be interpreted as control characters in json format. like \n, \t, \r etc.
"""

prompt_enhance_template = """
You'll be provided with a prompt that is intended to generate an AI image of a scene from a story. and a visual theme 
for the scene.
Enhance the given prompt to satisfy the constraints laid out in the Instructions.
Instructions:
--------------
    - You should stick to the given visual theme and the scene description.
    - You should intensify the wording and description so that the prompt leads us to a better quality AI generation.
    - If there's any ambiguity in the prompt and the objects mentioned, you should clarify with more detailed description.
    - The prompt should be detailed enough to generate a high-quality image. 
    - The prompt should be concise enough to be processed fully by a stable diffusion.
    - The prompt shouldn't get longer than 75 ish words. 
    - Use vivid language to create a compelling image.
--------------

**Input**
<Visual Theme>:
{visual_theme}

<original prompt>:
{original_prompt}

{format_instructions}
"""


prompt_enhance_template = PromptTemplate(prompt_enhance_template)
output_fixing_template = PromptTemplate(output_fixing_template)
rewrite_story_template = PromptTemplate(rewrite_story_template)
story_brief_template = PromptTemplate(story_brief_template)
story_template = PromptTemplate(story_template)
story_board_template = PromptTemplate(story_board_template)
scene_prompts_template = PromptTemplate(scene_prompts_template)
