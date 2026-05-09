from concurrent.futures import ThreadPoolExecutor
from typing import List


def _nori_analyze(client, index_name: str, query: str):
    """Analyze query with nori_custom (DCR) analyzer. Returns token set."""
    response = client.indices.analyze(
        index=index_name,
        body={"analyzer": "nori_custom", "text": query},
    )
    tokens = [t["token"] for t in response.get("tokens", [])]
    return " ".join(tokens), set(tokens)


def _term_count(client, index_name: str, doc_id: str, tokens: set) -> int:
    """Sum of term frequencies for query tokens in the doc's keywords field."""
    tv = client.termvectors(index=index_name, id=doc_id, fields=["keywords"], term_statistics=True)
    terms = tv.get("term_vectors", {}).get("keywords", {}).get("terms", {})
    return sum(terms.get(tok, {}).get("term_freq", 0) for tok in tokens)


# ── V2: keyword match + knn ────────────────────────────────────────────────────
def search_v2(client, index_name: str, query: str, query_vector: List[float], size: int = 5) -> List[dict]:
    body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {"match": {"keywords": {"query": query, "boost": 0.5}}},
                    {"knn": {"embedding": {"vector": query_vector, "k": size, "boost": 0.5}}},
                ]
            }
        },
    }
    return client.search(index=index_name, body=body)["hits"]["hits"]


# ── V3: vector search as base, term count (DCR keywords) as re-rank boost ─────
def search_v3(client, index_name: str, query: str, query_vector: List[float], size: int = 5) -> List[dict]:
    analyzed_query, tokens = _nori_analyze(client, index_name, query)

    # Run vector search and keyword term-vector fetch in parallel
    with ThreadPoolExecutor() as executor:
        knn_future = executor.submit(
            lambda: client.search(index=index_name, body={
                "size": size * 2,
                "query": {"knn": {"embedding": {"vector": query_vector, "k": size * 2}}},
            })["hits"]["hits"]
        )
        knn_hits = knn_future.result()

    # For each candidate from vector search: score = vector_score + term_count
    for hit in knn_hits:
        tc = _term_count(client, index_name, hit["_id"], tokens)
        hit["_combined_score"] = hit["_score"] + tc

    knn_hits.sort(key=lambda h: h["_combined_score"], reverse=True)
    return knn_hits[:size]
