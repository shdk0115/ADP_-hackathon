from typing import Dict


DEFAULT_ANALYSIS_RESULT: Dict = {
    "index_version": "v2",
    "use_keyword": True,
    "use_vector": True,
    "use_hybrid": True,
    "use_analyzer": True,
    "use_dcr": False,
    "fields": {
        "meta_data": "object",
        "contents": "text",
        "keywords": "text",
        "embedding": "knn_vector",
    },
    "analyzer_config": {"type": "nori", "decompound_mode": "mixed"},
}


def analyze_data_with_retry(user_prompt: str, max_retry: int = 3) -> Dict:
    # Placeholder for LLM-based analysis; returns deterministic default for now.
    _ = user_prompt
    _ = max_retry
    return DEFAULT_ANALYSIS_RESULT.copy()
