from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from config import SEARCH_K


def _nori_analyze(client, index_name: str, query: str):
    """Analyze query with nori_custom (DCR) analyzer. Returns (analyzed_str, token_set)."""
    response = client.indices.analyze(
        index=index_name,
        body={"analyzer": "nori_custom", "text": query},
    )
    tokens = [t["token"] for t in response.get("tokens", [])]
    return " ".join(tokens), set(tokens)


def _get_term_count(client, index_name: str, doc_id: str, tokens: set) -> int:
    tv = client.termvectors(index=index_name, id=doc_id, fields=["keyword"], term_statistics=True)
    terms = tv.get("term_vectors", {}).get("keyword", {}).get("terms", {})
    return sum(terms.get(tok, {}).get("term_freq", 0) for tok in tokens)


def _outer_join(text_hits: List[dict], vector_hits: List[dict]) -> List[dict]:
    seen = {}
    for hit in text_hits + vector_hits:
        if hit["_id"] not in seen:
            seen[hit["_id"]] = hit
    return list(seen.values())


def _normalize_scores(hits: List[dict]) -> List[dict]:
    scores = [h["_source"].get("combined_score", 0) for h in hits]
    mn, mx = min(scores, default=0), max(scores, default=1)
    rng = mx - mn or 1
    for hit in hits:
        raw = hit["_source"].get("combined_score", 0)
        hit["_source"]["normalized_score"] = round((raw - mn) / rng * 100, 2)
    return hits


# ── V2: keyword must match (min_match=1) + knn ───────────────────────────────
def search_v2(client, index_name: str, query: str, query_vector: List[float], size: int = SEARCH_K, text_min_score: int = 1) -> List[dict]:
    with ThreadPoolExecutor() as executor:
        text_future = executor.submit(
            lambda: client.search(index=index_name, body={
                "size": size * 2,
                "query": {
                    "bool": {
                        "must": [{
                            "match": {
                                "keyword": {
                                    "query": query,
                                    "minimum_should_match": text_min_score,
                                }
                            }
                        }]
                    }
                },
            })["hits"]["hits"]
        )
        vector_future = executor.submit(
            lambda: client.search(index=index_name, body={
                "size": size * 2,
                "query": {"knn": {"embedding": {"vector": query_vector, "k": size * 2}}},
            })["hits"]["hits"]
        )
        text_hits   = text_future.result()
        vector_hits = vector_future.result()

    vector_scores: Dict[str, float] = {h["_id"]: h["_score"] for h in vector_hits}
    combined = _outer_join(text_hits, vector_hits)
    for hit in combined:
        hit["_source"]["combined_score"] = hit["_score"] + vector_scores.get(hit["_id"], 0)

    combined = _normalize_scores(combined)
    combined.sort(key=lambda h: h["_source"]["combined_score"], reverse=True)
    return combined[:size]


# ── V3: parallel text(DCR+min_match=1) + vector, outer join, term count, normalize ──
def search_v3(
    client,
    index_name: str,
    query: str,
    query_vector: List[float],
    size: int = SEARCH_K,
    text_min_score: int = 1,
) -> List[dict]:
    analyzed_query, tokens = _nori_analyze(client, index_name, query)
    print(f"  [V3] 원문 질문: {query}")
    print(f"  [V3] DCR 분석 질문: {analyzed_query}")

    with ThreadPoolExecutor() as executor:
        text_future = executor.submit(
            lambda: client.search(index=index_name, body={
                "size": size * 2,
                "query": {
                    "bool": {
                        "must": [{
                            "match": {
                                "keyword": {
                                    "query": analyzed_query,
                                    "minimum_should_match": text_min_score,
                                }
                            }
                        }]
                    }
                },
            })["hits"]["hits"]
        )
        vector_future = executor.submit(
            lambda: client.search(index=index_name, body={
                "size": size * 2,
                "query": {"knn": {"embedding": {"vector": query_vector, "k": size * 2}}},
            })["hits"]["hits"]
        )
        text_hits   = text_future.result()
        vector_hits = vector_future.result()

    # Term count from text results
    text_term_counts: Dict[str, int] = {
        hit["_id"]: _get_term_count(client, index_name, hit["_id"], tokens)
        for hit in text_hits
    }
    vector_scores: Dict[str, float] = {h["_id"]: h["_score"] for h in vector_hits}

    # Outer join → combined score
    combined = _outer_join(text_hits, vector_hits)
    for hit in combined:
        doc_id = hit["_id"]
        tc = text_term_counts.get(doc_id, 0)
        vs = vector_scores.get(doc_id, 0)
        hit["_source"]["term_count"]     = tc
        hit["_source"]["vector_score"]   = vs
        hit["_source"]["combined_score"] = tc + vs

    combined = _normalize_scores(combined)
    combined.sort(key=lambda h: h["_source"]["combined_score"], reverse=True)

    print(f"  [V3] Hybrid Search 완료 (결과={len(combined[:size])}건)")
    return combined[:size]
