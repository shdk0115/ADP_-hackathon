import json
import re
import boto3
from typing import List, Optional
from config import AWS_REGION, BEDROCK_LLM_MODEL


def _call_llm(prompt: str) -> str:
    print(f"  🤖 LLM 호출 [hyde] ({len(prompt)}자)")
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(
        modelId=BEDROCK_LLM_MODEL, body=body,
        contentType="application/json", accept="application/json",
    )
    return json.loads(response["body"].read())["content"][0]["text"].strip()


def summarize_text(text: str) -> str:
    prompt = (
        f"Summarize the following text concisely in the same language.\n"
        f"Return ONLY the summary, no explanation.\n\nTEXT:\n{text}"
    )
    try:
        return _call_llm(prompt)
    except Exception:
        return ""


def _qa_question(qa) -> str:
    """Extract a question string from a qa item that may be a dict or a plain str."""
    if isinstance(qa, dict):
        return str(qa.get("question") or qa.get("질문") or qa.get("Q") or "").strip()
    if isinstance(qa, str):
        return qa.strip()
    return ""


def match_questions(text: str, qa_sheet: List[dict]) -> List[str]:
    if not qa_sheet:
        return []
    items = []
    for i, qa in enumerate(qa_sheet):
        q = _qa_question(qa)
        if q:
            items.append({"id": i, "question": q})
    if not items:
        return []
    questions_json = json.dumps(items, ensure_ascii=False)
    prompt = (
        f"Given the TEXT below, return the IDs of questions from the list that this text can answer.\n"
        f"Return ONLY a JSON array of IDs, e.g. [0, 2, 5]. No explanation.\n\n"
        f"TEXT:\n{text}\n\nQUESTIONS:\n{questions_json}"
    )
    try:
        raw = _call_llm(prompt)
        ids = json.loads(re.search(r"\[.*?\]", raw, re.DOTALL).group())
        out = []
        for i in ids:
            if 0 <= i < len(qa_sheet):
                q = _qa_question(qa_sheet[i])
                if q:
                    out.append(q)
        return out
    except Exception:
        return []


def build_hyde_content(text: str, qa_sheet: List[dict]) -> str:
    """
    Returns: TEXT | summary | matched questions
    This is the full string to embed for V3.
    """
    summary = summarize_text(text)
    questions = match_questions(text, qa_sheet)

    parts = [text]
    if summary:
        parts.append(summary)
    if questions:
        parts.append(" | ".join(questions))
    return " | ".join(parts)
