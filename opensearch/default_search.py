from config import SEARCH_K

def default_hybrid_search(client, index_name, query, query_vector, size=SEARCH_K):
    body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "TEXT": {
                                "query": query,
                                "boost": 0.5
                            }
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": size,
                                "boost": 0.5
                            }
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index=index_name, body=body)
    return response["hits"]["hits"]
