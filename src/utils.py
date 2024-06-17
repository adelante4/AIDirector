import streamlit as st


def add_comment():
    st.session_state.comments.append('')


def handle_submit():
    post_description = st.session_state.post_description_input
    comments = st.session_state.comments_input

    if not post_description:
        st.error("Please fill in the post description before proceeding.")
        st.stop()
    for idx, comment in enumerate(comments):
        if not comment:
            st.error(f"Please fill in comment {idx + 1} before proceeding.")
            st.stop()

    st.session_state.post_description = post_description
    st.session_state.comments = comments
