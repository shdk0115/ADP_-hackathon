from langchain.llms import Bedrock
from langchain.embeddings import BedrockEmbeddings
from config import AWS_REGION, BEDROCK_TEXT_MODEL, BEDROCK_EMBED_MODEL


def get_text_llm():
    return Bedrock(
        region_name=AWS_REGION,
        model_id=BEDROCK_TEXT_MODEL
    )


def get_embedding_model():
    return BedrockEmbeddings(
        region_name=AWS_REGION,
        model_id=BEDROCK_EMBED_MODEL
    )
