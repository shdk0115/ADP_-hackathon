from typing import Callable, Dict, List, Optional
import re


def build_hyde_query(user_query: str, domain_hint: Optional[str] = None) -> str:
    hint = f" Domain: {domain_hint}." if domain_hint else ""
    return (
        f"Hypothetical answer document for retrieval.{hint} "
        f"Question: {user_query} "
        f"Answer draft: This document explains key facts, entities, and procedures relevant to the question."
    )


def summarize_row_for_hyde(row: Dict, llm_summary_fn: Optional[Callable[[str], str]] = None) -> str:
    base_text = (
        f"카테고리 {row.get('category', '')}. "
        f"상품명 {row.get('product', '')}. "
        f"가격 {row.get('price', '')}. "
        f"평점 {row.get('rating', '')}. "
        f"해시태그 {row.get('hashtags', '')}. "
        f"추천상황 {row.get('use_case', '')}."
    ).strip()
    if llm_summary_fn is not None:
        return llm_summary_fn(base_text)
    # Fallback summary template when LLM client is not connected.
    return (
        f"{row.get('product', '')}는 {row.get('category', '')} 카테고리의 상품으로, "
        f"가격은 {row.get('price', '')}, 평점은 {row.get('rating', '')}이며 "
        f"{row.get('use_case', '')} 상황에 적합하다."
    )


def select_related_questions(row: Dict, qa_sheet: List[Dict], top_k: int = 3) -> List[str]:
    product = row.get("product", "").lower()
    category = row.get("category", "").lower()
    hashtags = row.get("hashtags", "").lower()
    tag_tokens = re.findall(r"[a-z0-9가-힣]+", hashtags)
    related: List[tuple] = []
    for qa in qa_sheet:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        ql = q.lower()
        score = 0
        if product and product in a.lower():
            score += 5
        if product and any(tok in ql for tok in re.findall(r"[a-z0-9가-힣]+", product)):
            score += 2
        if category and category in ql:
            score += 1
        score += sum(1 for tok in tag_tokens if tok and tok in ql)
        if score > 0:
            related.append((score, q))
    related.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in related[:top_k]]


def build_hyde_document(row: Dict, qa_sheet: List[Dict], llm_summary_fn: Optional[Callable[[str], str]] = None) -> str:
    summary = summarize_row_for_hyde(row, llm_summary_fn=llm_summary_fn)
    questions = select_related_questions(row, qa_sheet, top_k=3)
    question_block = " | ".join(questions) if questions else ""
    return f"hyde_summary: {summary} hyde_related_questions: {question_block}".strip()
