import streamlit as st

from src.data.models.api_models import RequestAPIObject
from src.director.director import Director
from src.utils import add_comment, handle_submit
from llama_index.core import Settings
from llama_index.llms.groq import Groq

st.title('AI Director')
st.write('Welcome to the AI Director! This tool will generate prompts building an animation for '
         'your given post and comments')

if 'comments' not in st.session_state:
    st.session_state.comments = ['']

post_description_input = st.text_area("Post Description", key="post_description_input")
api_key = st.text_input('Groq API Key', '')
model_name = st.text_input('Model Name', 'llama3-70b-8192')


st.write("Comments:")
comments_input = []
for i, comment in enumerate(st.session_state.comments):
    comment_input = st.text_input(f"Comment {i + 1}", value=comment, key=f"comment_{i}")
    comments_input.append(comment_input)

st.session_state.comments_input = comments_input

st.button("Add Comment", on_click=add_comment)

st.button("Generate Prompts", on_click=handle_submit)

if 'post_description' in st.session_state:
    placeholder = st.empty()
    progress_bar = st.progress(0)

    Settings.llm = Groq(model=model_name, api_key=api_key)
    director = Director(RequestAPIObject(post=st.session_state.post_description,
                                         comments=st.session_state.comments,
                                         groq_api_key=api_key,
                                         groq_model_name=model_name,
                                         progress_bar=progress_bar))
    scene_prompts = director.build_animation()
    st.write(scene_prompts)
