from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import Bedrock
import re
from config import AWS_REGION, BEDROCK_TEXT_MODEL
from bedrock_client import get_text_llm

def get_keyword_extraction_template():
    return """
    You are a keyword extractor. Analyze the given <text> and follow the rules below.

    Rules:
    - Extract ONLY important keywords from the text below.
    - Return them as a space-separated list.
    - No explanations. 
    - No punctuation.
    - <Output> Should ONLY consist of keywords.

    <text>
    {text}
    </text>

    <Output>
    Output:
    <Output>
    """


def build_keyword_extractor():
    llm = get_text_llm()
    prompt = PromptTemplate(
        input_variables=["text"],
        template=get_keyword_extraction_template()
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def extract_keywords(text, chain):
    raw_output = chain.invoke({"text": text})

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
