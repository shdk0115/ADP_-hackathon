from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import Bedrock
import re
from config import AWS_REGION, BEDROCK_TEXT_MODEL


def get_keyword_extraction_template():
    return """
    You are a keyword extractor.

    Extract ONLY important keywords from the text below.
    Return them as a space-separated list.
    No explanations. No punctuation.

    <text>
    {text}
    </text>
    """


def build_keyword_extractor(region=AWS_REGION, model_id=BEDROCK_TEXT_MODEL):
    llm = Bedrock(
        region_name=region,
        model_id=model_id
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template=get_keyword_extraction_template()
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def extract_keywords(text, chain):
    raw_output = chain.invoke({"review": text})

    # Clean + normalize
    keywords = re.findall(r'\b\w+\b', raw_output.lower())

    # dedupe but keep order
    seen = set()
    result = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            result.append(w)

    return result
