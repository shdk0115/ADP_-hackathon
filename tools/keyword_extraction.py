import json
import re
import boto3
from typing import List
from config import AWS_REGION, BEDROCK_LLM_MODEL


def extract_keywords(text: str) -> List[str]:
    print(f"  🤖 LLM 호출 [extract_keywords] ({len(text)}자)")
    prompt = (
        f"Extract all significant keywords from the text below.\n"
        f"Return ONLY the keywords delimited by whitespace. No explanation, no punctuation.\n\n"
        f"TEXT: {text}"
    )

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    if "claude" in BEDROCK_LLM_MODEL.lower() or "anthropic" in BEDROCK_LLM_MODEL.lower():
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(modelId=BEDROCK_LLM_MODEL, body=body,
                                       contentType="application/json", accept="application/json")
        raw = json.loads(response["body"].read())["content"][0]["text"].strip()
    else:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": 512, "temperature": 0.0},
        })
        response = client.invoke_model(modelId=BEDROCK_LLM_MODEL, body=body,
                                       contentType="application/json", accept="application/json")
        raw = json.loads(response["body"].read())["results"][0]["outputText"].strip()

    return re.findall(r"[A-Za-z0-9가-힣_]+", raw)
