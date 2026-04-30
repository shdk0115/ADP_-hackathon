from typing import Dict, List

from opensearch.agentic_index import create_agent_index
from opensearch.ingest import ingest_documents_default
from tools.keyword_extraction import extract_keywords


def enrich_docs_for_agent_index(documents: List[Dict], use_keyword: bool) -> List[Dict]:
    enriched = []
    for doc in documents:
        row = dict(doc)
        if use_keyword:
            row["keywords"] = " ".join(extract_keywords(row.get("contents", "")))
        enriched.append(row)
    return enriched


def run_index_agent(client, index_name: str, analysis_result: Dict, documents: List[Dict]) -> None:
    create_agent_index(client, index_name, analysis_result)
    docs = enrich_docs_for_agent_index(documents, analysis_result.get("use_keyword", True))
    ingest_documents_default(client, index_name, docs)
