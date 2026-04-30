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


def create_agent_index(client, index_name: str, analysis_result: dict):
    version = analysis_result.get("index_version", "v2")
    use_analyzer = analysis_result.get("use_analyzer_dcr", analysis_result.get("use_analyzer", False))

    properties = {
        "meta_data": {"type": "object"},
        "contents": {"type": "text"},
        "embedding": {
            "type": "knn_vector",
            "dimension": EMBEDDING_CONFIG["dimension"],
            "similarity": EMBEDDING_CONFIG["similarity"],
        },
    }

    if version in {"v2", "v3"}:
        properties["keywords"] = {"type": "text"}

    body = {"settings": _base_settings(), "mappings": {"properties": properties}}

    if use_analyzer:
        decompound_mode = (
            analysis_result.get("analyzer_config", {}).get("decompound_mode", "mixed")
        )
        body["settings"].update(build_nori_analyzer_config(decompound_mode=decompound_mode))

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=body)
