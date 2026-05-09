from config import OPENSEARCH_DEFAULTS, EMBEDDING_CONFIG


def create_index_v1(client, index_name="documents"):
    if not client.indices.exists(index=index_name):
        body = {
            "settings": {
                "index": {
                    "number_of_shards": OPENSEARCH_DEFAULTS["number_of_shards"],
                    "number_of_replicas": OPENSEARCH_DEFAULTS["number_of_replicas"],
                    "knn": OPENSEARCH_DEFAULTS["knn"]
                }
            },
            "mappings": {
                "properties": {
                    "meta_info": {
                        "type": "object"
                    },
                    "TEXT": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_CONFIG["dimension"],
                        "similarity": EMBEDDING_CONFIG["similarity"]
                    }
                }
            }
        }

        client.indices.create(index=index_name, body=body)
        print(f"✅ Created index: {index_name}")
    else:
        print(f"⚠️ Index already exists: {index_name}")
