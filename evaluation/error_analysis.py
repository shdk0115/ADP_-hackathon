from collections import Counter
from typing import Dict, List


def analyze_errors(errors: List[Dict]) -> Dict:
    if not errors:
        return {"count": 0, "by_question_length": {}, "notes": ["No errors found."]}

    buckets = Counter()
    for row in errors:
        q_len = len(row.get("question", ""))
        if q_len < 20:
            buckets["short"] += 1
        elif q_len < 60:
            buckets["medium"] += 1
        else:
            buckets["long"] += 1

    return {
        "count": len(errors),
        "by_question_length": dict(buckets),
        "notes": [
            "Review top_text vs expected mismatch.",
            "Check whether keyword extraction and analyzer should be enabled.",
        ],
    }
