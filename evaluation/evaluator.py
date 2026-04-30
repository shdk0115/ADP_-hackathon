from typing import Callable, Dict, List
import time


def _get_first_hit_text(hits: List[Dict]) -> str:
    if not hits:
        return ""
    return hits[0].get("_source", {}).get("contents", "")


def evaluate_qa_sheet(
    qa_sheet: List[Dict],
    search_fn: Callable[[str], List[Dict]],
) -> Dict:
    if not qa_sheet:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "errors": [],
            "mean_search_ms": 0.0,
        }

    correct = 0
    errors = []
    total_search_ms = 0.0

    for item in qa_sheet:
        question = item.get("question", "")
        expected = item.get("answer", "")
        start = time.perf_counter()
        hits = search_fn(question)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_search_ms += elapsed_ms
        top_text = _get_first_hit_text(hits).lower()
        ok = expected.lower() in top_text if expected else bool(hits)
        if ok:
            correct += 1
        else:
            errors.append({"question": question, "expected": expected, "top_text": top_text})

    total = len(qa_sheet)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "errors": errors,
        "mean_search_ms": round(total_search_ms / total, 3),
    }
