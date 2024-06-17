from llama_index.core import PromptTemplate

story_template = (
    """
    Imagine you're a very dark, experienced writer with a very absurd view of life. You are an avant-garde artist
    whose works are always different from his generation. The genre of your works can be described as 
    post-modern inspired by the living situations.
 
    For your new story, You'll be given one social media post with some people's comments. 
    Your task is to extract an innovative short story with the main ideas as the whole discussion.
    
    **Instructions**
    1. First extract the very abstract subject or a meaning for the whole conversation. 
    2. Categorize the users who commented in Sociological groups. 
    3. Try to reflect on the interaction between such characters in your story.
    4. The short story should be around 2-3 paragraphs long.
    5. Get creative with it. Remember what kind of writer you are.
    6. Use fictional characters instead of each group.
    7. The subject of your story should not be very similar to the news, the similarity should be in the deeper, abstract levels of the story.
    8. The same should apply to the comments, you should not mention exactly the same stuff, you should just be inspired by them.
    9. You don't have to use all of the comments.
    
    You'll  be given the information in the following format:
    <Post>: the header
    
    <Comment>: some comment from a user
    <Comment>: some comment from another user
    ...
    
    **Input**
    <Post>: {post}
    
    {comments}
    
    ONLY provide the json, no extra words around it.
    """
)

story_board_template = (
    """
    Imagine you're a very artistic short-animation director, who works in a post-modern, avant-garde, surreal genre. 
    You'll be provided with a short story, your task is to make a storyboard for this story. 
    
    **Instructions**
    1. Divide the story into as many cinematic scenes as needed. 
    2. Describe each scene in a way that it's straightforward enough for your illustrator to draw them. 
    3. Add a short note for the art director to describe the whole visual theme of the animation.
    
    **Input Story**
    {story}
    
    ONLY provide the json, no extra words around it.
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
    """
)


story_template = PromptTemplate(story_template)
story_board_template = PromptTemplate(story_board_template)
scene_prompts_template = PromptTemplate(scene_prompts_template)
