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
                "embedding": doc.get("embedding", []),
                "keywords": doc.get("keywords", ""),
            }
        }
        actions.append(action)

    helpers.bulk(client, actions)
    print(f"✅ Ingested {len(actions)} documents into {index_name}")

# List Index 
def list_index_mappings(client, index_name):
    # OpenSearch 클러스터에서 인덱스 목록 가져오기
    indices = client.indices.get_alias().keys()

    # 인덱스 목록 출력
    print("Index List:")
    for index in indices:
        print(index)
        
    print('='*100)
    # 인덱스 유/무 확인
    if index_name not in indices:
        print(f"Index '{index_name}' Does Not Exist")
        return  # Exit the function if index_name does not exist
    
    # 매핑 출력
    mapping = client.indices.get_mapping(index=index_name)
    print('='*100)
    print("Mapping for index '{}':".format(index_name))
    print(mapping)

# Check Ingest Count
def check_ingest_count(user_info, query_size, index_name):
    import time
    if len(user_info.indices.get(index_name)) == 1:
        search_query = {
            "query" : {
                "match_all" : {}
            }
        }
        start = time.time()

        results = user_info.count(index=index_name, body = search_query)
        end = time.time()

        print(results)
        print(f"{end - start} 소요")
    else:
         print("Index is Not Found in Your Opensearch") 
