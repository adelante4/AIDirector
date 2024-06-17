from typing import Optional
from streamlit.delta_generator import DeltaGenerator
from pydantic import BaseModel, Field


class RequestAPIObject(BaseModel):
    post: str = Field(description="the social media post description")
    comments: list[str] = Field(description="The list of comments on the post")
    groq_model_name: str = Field(description="The name of the model to be used", default="llama3-70b-8192")
    groq_api_key: str = Field(description="The API key for the Groq model")
    progress_bar: Optional[DeltaGenerator] = Field(description="The progress bar")

    class Config:
        arbitrary_types_allowed = True
