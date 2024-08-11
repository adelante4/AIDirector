import logging

from llama_index.core import Document

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from settings import MAX_ATTEMPTS, EMBEDDING_MODEL_NAME
from src.data.models.response_models import StoryBriefOutput
from src.utils.fixing_output_parser import FixingOutputParser
from src.director.constants import acclaimed_controversial_stories, acclaimed_controversial_stories_briefs
from src.director.director import Director
from src.director.prompts import story_brief_template


class StorySamplesRetriever:

    @staticmethod
    def get_briefs() -> dict:
        """
        Get briefs for the acclaimed and controversial stories.
        :return:
        dict: {story: StoryBriefOutput}
        """
        brief_program = Director.get_completion_program(
            output_parser=FixingOutputParser(StoryBriefOutput),
            prompt_template_str=story_brief_template.get_template(),
            output_cls=StoryBriefOutput,
        )

        current_attempt = 0
        briefs = {}
        briefs_done = {}

        while (len(briefs) < len(acclaimed_controversial_stories)
               and current_attempt < MAX_ATTEMPTS):
            for story in acclaimed_controversial_stories:
                if story in briefs_done.keys():
                    continue
                try:
                    briefs[story] = brief_program(story=story)
                except Exception as e:
                    logging.error(f"Error for {story}: {e}")
                    briefs[story] = None
            briefs_done = {k: v for k, v in briefs.items() if v is not None}
            current_attempt += 1

        return briefs_done

    def generate_documents(self) -> list:
        """
        Get documents for the briefs.
        :return: 
        list: List of Document objects
        """
        logging.info("Generating the briefs...")
        briefs = self.get_briefs()

        documents = []
        for story, brief in briefs.items():
            documents.append(Document(text=brief.brief + " " + brief.story_analysis + " " + brief.artistic_style,
                                      metadata={"story_analysis": brief.story_analysis,
                                                "artistic_style": brief.artistic_style, "story": story}))

        return documents

    def get_retriever(self, rebuild_retriever: bool = False):
        """
        Get the retriever object for the story samples.
        :param rebuild_retriever:         
        """
        if rebuild_retriever:
            documents = self.generate_documents()
        else:
            documents = self.load_documents()

        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.get_or_create_collection("stories")

        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5
        )

        return retriever

    @staticmethod
    def load_documents():
        documents = []
        for story, brief in acclaimed_controversial_stories_briefs.items():
            documents.append(Document(text=brief['brief'] + " " + brief['story_analysis'] + " " + brief['artistic_style'],
                                      metadata={"story_analysis": brief['story_analysis'],
                                                "artistic_style": brief['artistic_style'], "story": story}))

        return documents
