import re
from collections import Counter
from typing import Dict, List

_DATE_PATTERN = re.compile(r"\d{4}년|\d+월|\d+일|이전|이후|기준|입사|근속")
_ENTITY_PATTERN = re.compile(r"유닛|팀|부서|조직|센터|본부")
_COND_PATTERN = re.compile(r"특례|예외|단,|단 |경우|조건|해당|적용")
_TABLE_PATTERN = re.compile(r"얼마|누구|몇|금액|한도|한도는|승인|전결")

def _detect_patterns(errors: List[Dict]) -> List[str]:
    patterns = []
    for e in errors:
        q = e.get("question", "")
        got = e.get("got", "") or e.get("top_text", "")
        exp = e.get("expected", "")

        if _DATE_PATTERN.search(q):
            patterns.append("DATE_BOUNDARY_FAIL")
        if _ENTITY_PATTERN.search(q):
            patterns.append("ENTITY_MAPPING_FAIL")
        if _COND_PATTERN.search(q):
            patterns.append("CONDITIONAL_REASONING_FAIL")
        if _TABLE_PATTERN.search(q) and got != exp:
            patterns.append("TABLE_RETRIEVAL_FAIL")
        if got and exp and exp.lower() not in got.lower():
            patterns.append("VECTOR_ONLY_INSUFFICIENT")

    return list(dict.fromkeys(patterns))  # 순서 유지 dedup

def analyze_errors(errors: List[Dict]) -> Dict:
    if not errors:
        return {"count": 0, "error_patterns": [], "notes": ["No errors."]}

    patterns = _detect_patterns(errors)
    return {
        "count": len(errors),
        "error_patterns": patterns,
        "failed_ids": [e.get("question","")[:30] for e in errors],
        "notes": [
            "Review top_text vs expected mismatch.",
            f"Detected patterns: {patterns}",
        ],
    }