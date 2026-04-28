from opensearchpy import helpers

def ingest_documents_default(client, index_name, documents):
    """
    documents: List[Dict]
    Example:
    [
        {
            "meta_data": {...},
            "contents": "text chunk",
            "embedding": [float vector]
        }
    ]
    """

    actions = []

    for i, doc in enumerate(documents):
        action = {
            "_index": index_name,
            "_id": doc.get("id", i),
            "_source": {
                "meta_data": doc.get("meta_data", {}),
                "contents": doc.get("contents", ""),
                "embedding": doc.get("embedding", [])
            }
        }
        actions.append(action)

    helpers.bulk(client, actions)
    print(f"✅ Ingested {len(actions)} documents into {index_name}")
