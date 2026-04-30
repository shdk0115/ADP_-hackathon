from typing import Iterable, List


def get_decompound_rules(keyword_rows: Iterable[Iterable[str]]) -> List[str]:
    rules = []
    seen = set()
    for row in keyword_rows:
        for keyword in row:
            if keyword not in seen:
                seen.add(keyword)
                rules.append(keyword)
    return rules
