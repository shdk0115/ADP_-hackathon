import json
from pathlib import Path
from typing import Dict, List

from agents.data_analysis import analyze_data_with_retry
from config import (
    MAX_ANALYSIS_RETRY,
    OPENSEARCH_HOST,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_USE_SSL,
)
from evaluation.error_analysis import analyze_errors
from evaluation.evaluator import evaluate_qa_sheet
from opensearch.agentic_index import create_agent_index
from opensearch.agentic_search import agent_hybrid_search
from opensearch.default_index import create_index_v1
from opensearch.default_search import default_hybrid_search
from opensearch.ingest import ingest_documents_default
from tools.keyword_extraction import extract_keywords
from tools.preprocessing import hash_embed, load_raw_documents, preprocess_agentic, preprocess_default

try:
    from opensearchpy import OpenSearch
except ImportError:  # pragma: no cover
    OpenSearch = None

BASE_DIR = Path(__file__).resolve().parent


def _load_qa_sheet(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists() or not p.read_text(encoding="utf-8").strip():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _get_client():
    if OpenSearch is None:
        raise RuntimeError("opensearchpy is not installed. Run: pip install opensearch-py")

    auth = (OPENSEARCH_USER, OPENSEARCH_PASSWORD) if OPENSEARCH_USER and OPENSEARCH_PASSWORD else None
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def _reset_index(client, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)


def _keyword_exposure_count(client, index_name: str, doc_id: str, query_tokens: List[str]) -> int:
    tv = client.termvectors(index=index_name, id=doc_id, fields=["keywords"], term_statistics=True)
    terms = tv.get("term_vectors", {}).get("keywords", {}).get("terms", {})
    return sum(terms.get(tok, {}).get("term_freq", 0) for tok in query_tokens)


def _search_v1(client, index_name: str, query: str, size: int = 5) -> List[Dict]:
    qv = hash_embed(query)
    return default_hybrid_search(client, index_name, query, qv, size=size)


def _search_v2(client, index_name: str, query: str, size: int = 5) -> List[Dict]:
    qv = hash_embed(query)
    hits = agent_hybrid_search(client, index_name, query, qv, use_keyword=True, size=size * 2)
    query_tokens = extract_keywords(query, top_k=10)
    for hit in hits:
        freq = _keyword_exposure_count(client, index_name, hit["_id"], query_tokens)
        hit["_score"] = float(hit.get("_score", 0.0)) + (1.5 * freq)
    hits.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return hits[:size]


def _search_v3(client, index_name: str, query: str, size: int = 5) -> List[Dict]:
    qv = hash_embed(query)
    normalized_query = "".join(query.split())
    hits = agent_hybrid_search(client, index_name, normalized_query, qv, use_keyword=True, size=size * 2)
    query_tokens = extract_keywords(query, top_k=10)
    for hit in hits:
        freq = _keyword_exposure_count(client, index_name, hit["_id"], query_tokens)
        hit["_score"] = float(hit.get("_score", 0.0)) + (1.5 * freq)
    hits.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return hits[:size]


def _print_terminal_results_table(eval_map: Dict[str, Dict]) -> None:
    print("\n=== Retrieval Evaluation Dashboard ===")
    header = f"{'Version':<10} {'Accuracy':<12} {'Correct/Total':<14} {'Mean Search ms':<16}"
    print(header)
    print("-" * len(header))
    for version in ("v1", "v2", "v3"):
        row = eval_map[version]
        acc = f"{row['accuracy']:.4f}"
        corr = f"{row['correct']}/{row['total']}"
        mean_ms = f"{row['mean_search_ms']:.3f}"
        print(f"{version:<10} {acc:<12} {corr:<14} {mean_ms:<16}")
    print("=" * len(header))


def run_opensearch_pipeline(
    user_prompt: str,
    raw_data_path: str = "data/raw_data",
    qa_sheet_path: str = "data/qa_sheet.json",
) -> Dict:
    raw_path = str((BASE_DIR / raw_data_path).resolve())
    qa_path = str((BASE_DIR / qa_sheet_path).resolve())
    raw_docs = load_raw_documents(raw_path)
    qa_sheet = _load_qa_sheet(qa_path)
    client = _get_client()

    default_docs = preprocess_default(raw_docs, embedding_fn=hash_embed)
    agentic_v2_docs = preprocess_agentic(
        raw_docs,
        keyword_fn=lambda x: extract_keywords(x, top_k=10),
        embedding_fn=hash_embed,
        use_analyzer_dcr=False,
        use_hyde=False,
        qa_sheet=qa_sheet,
    )
    agentic_v3_docs = preprocess_agentic(
        raw_docs,
        keyword_fn=lambda x: extract_keywords(x, top_k=10),
        embedding_fn=hash_embed,
        use_analyzer_dcr=True,
        use_hyde=True,
        qa_sheet=qa_sheet,
    )

    analysis = analyze_data_with_retry(user_prompt, max_retry=MAX_ANALYSIS_RETRY)

    v1_index = "retail_v1_eval"
    v2_index = "retail_v2_eval"
    v3_index = "retail_v3_eval"

    _reset_index(client, v1_index)
    create_index_v1(client, v1_index)
    ingest_documents_default(client, v1_index, default_docs)
    client.indices.refresh(index=v1_index)
    v1_eval = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_v1(client, v1_index, q))

    _reset_index(client, v2_index)
    create_agent_index(
        client,
        v2_index,
        {
            **analysis,
            "index_version": "v2",
            "use_analyzer_dcr": False,
            "use_analyzer": False,
        },
    )
    ingest_documents_default(client, v2_index, agentic_v2_docs)
    client.indices.refresh(index=v2_index)
    v2_eval = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_v2(client, v2_index, q))

    _reset_index(client, v3_index)
    create_agent_index(
        client,
        v3_index,
        {
            **analysis,
            "index_version": "v3",
            "use_analyzer_dcr": True,
            "use_analyzer": True,
        },
    )
    ingest_documents_default(client, v3_index, agentic_v3_docs)
    client.indices.refresh(index=v3_index)
    v3_eval = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_v3(client, v3_index, q))

    err = analyze_errors(v3_eval["errors"])
    eval_map = {"v1": v1_eval, "v2": v2_eval, "v3": v3_eval}
    _print_terminal_results_table(eval_map)

    return {
        "v1_eval": v1_eval,
        "v2_eval": v2_eval,
        "v3_eval": v3_eval,
        "analysis_result": analysis,
        "error_analysis": err,
    }


if __name__ == "__main__":
    try:
        result = run_opensearch_pipeline("사용자 프롬프트 + 데이터 분석 + QA 평가")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print("OpenSearch evaluation failed.")
        print(f"Reason: {exc}")
        print(
            "Check OPENSEARCH_HOST/PORT/USER/PASSWORD in .env and ensure the cluster is running."
        )
