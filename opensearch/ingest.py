from opensearchpy import helpers

def _bulk_ingest(client, index_name, actions):
    helpers.bulk(client, actions)
    print(f"✅ Ingested {len(actions)} documents into {index_name}")

def ingest_documents_v1(client, index_name, documents):
    """V1: meta_info/content/embedding → stored as meta_info/TEXT/embedding."""
    actions = [
        {
            "_index": index_name,
            "_id": doc.get("id", i),
            "_source": {
                "meta_info": doc.get("meta_info", {}),
                "TEXT": doc.get("content", ""),
                "embedding": doc.get("embedding", []),
            }
        }
        for i, doc in enumerate(documents)
    ]
    _bulk_ingest(client, index_name, actions)

def ingest_documents_v2(client, index_name, documents):
    """V2: read preprocess_agentic output (metainfo/TEXT/keyword/embedding).
    OS에는 TEXT 와 keyword 둘 다 '본문+키워드+DCR' 통합 텍스트로 저장 →
    search_v2(match: keyword) / evaluator(read: TEXT) 둘 다 만족."""
    actions = []
    for i, doc in enumerate(documents):
        text = (doc.get("TEXT", "") or "").strip()
        keyword = (doc.get("keyword", "") or "").strip()
        merged = f"{text} {keyword}".strip() if (text or keyword) else ""
        actions.append({
            "_index": index_name,
            "_id": doc.get("id", i),
            "_source": {
                "meta_info": doc.get("metainfo", {}),
                "TEXT": merged,
                "keyword": merged,
                "embedding": doc.get("embedding", []),
            }
        })
    _bulk_ingest(client, index_name, actions)

def ingest_documents_v3(client, index_name, documents):
    """V3: same fields as V2 — search_v3 uses knn + termvectors(keyword) re-rank."""
    ingest_documents_v2(client, index_name, documents)

def ingest_documents_default(client, index_name, documents):
    """Legacy wrapper — delegates to v1."""
    ingest_documents_v1(client, index_name, documents)

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
