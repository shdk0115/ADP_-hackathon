import re
from collections import Counter
from typing import List

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "have",
    "will",
    "your",
    "you",
}


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9가-힣_]+", text.lower())
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(top_k)]
