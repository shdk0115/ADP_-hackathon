from config import EMBEDDING_CONFIG, OPENSEARCH_DEFAULTS
from tools.analyzer import build_nori_analyzer_config


def _base_settings():
    return {
        "index": {
            "number_of_shards": OPENSEARCH_DEFAULTS["number_of_shards"],
            "number_of_replicas": OPENSEARCH_DEFAULTS["number_of_replicas"],
            "knn": OPENSEARCH_DEFAULTS["knn"],
        }
    }


def create_agent_index(client, index_name: str, analysis_result: dict, dcr_rules: list = None):
    version = analysis_result.get("index_version", "v2")
    use_analyzer = analysis_result.get("use_analyzer_dcr", analysis_result.get("use_analyzer", False))

    text_field = {"type": "text", "analyzer": "nori_custom"} if use_analyzer else {"type": "text"}
    keyword_field = {"type": "text", "analyzer": "nori_custom"} if use_analyzer else {"type": "text"}

    properties = {
        "meta_info": {"type": "object"},
        "TEXT": text_field,
        "keyword": keyword_field,
        "embedding": {
            "type": "knn_vector",
            "dimension": EMBEDDING_CONFIG["dimension"],
            "similarity": EMBEDDING_CONFIG["similarity"],
        },
    }

    body = {"settings": _base_settings(), "mappings": {"properties": properties}}

    if use_analyzer:
        decompound_mode = analysis_result.get("analyzer_config", {}).get("decompound_mode", "mixed")
        analyzer_settings = build_nori_analyzer_config(
            decompound_mode=decompound_mode,
            dcr_rules=dcr_rules or [],
        )
        body["settings"].update(analyzer_settings)

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=body)
