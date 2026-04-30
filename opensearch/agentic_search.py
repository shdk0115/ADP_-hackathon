from concurrent.futures import ThreadPoolExecutor
from typing import List


def agent_hybrid_search(
    client,
    index_name: str,
    query_text: str,
    query_vector: List[float],
    use_keyword: bool = True,
    size: int = 5,
):
    should = []
    if use_keyword:
        should.append({"match": {"keywords": {"query": query_text, "boost": 0.4}}})
    should.append({"match": {"contents": {"query": query_text, "boost": 0.3}}})
    should.append(
        {"knn": {"embedding": {"vector": query_vector, "k": size, "boost": 0.3}}}
    )

    body = {"size": size, "query": {"bool": {"should": should}}}
    response = client.search(index=index_name, body=body)
    return response["hits"]["hits"]


def select_custom_hybrid_query_w_score_parallel(
    client,
    question,
    index_name,
    query_size,
    filter_word,
    threshold_score=0,
    text_min_score=1,
):
    """
    NOTE: This function preserves the existing team hybrid-search strategy shape.
    Integrate project-specific implementations for nori/token/embedding/text/vector search.
    """
    analyzer = "nori_analyzer_w_dcr"
    analyzed_question, tokens = nori_analyze_w_token(client, index_name, analyzer, question)
    print("Text Search Input Question:", question)
    print("Hybrid Search Input Question:", analyzed_question)

    region = "us-east-1"
    embedding_model_id = "amazon.titan-embed-text-v1"
    embedded_query = text_embedding_with_bedrock(question, region, embedding_model_id)

    with ThreadPoolExecutor() as executor:
        text_search_future = executor.submit(
            text_search,
            client,
            analyzed_question,
            index_name,
            query_size,
            filter_word,
            text_min_score,
        )
        vector_search_future = executor.submit(
            vector_search,
            client,
            embedded_query,
            index_name,
            query_size,
        )
        text_search_results = text_search_future.result()
        vector_search_results = vector_search_future.result()

    text_results_dict = {}
    for hit in text_search_results["hits"]["hits"]:
        doc_id = hit["_id"]
        table_nm_kor = hit["_source"]["table_name_kor"]
        table_keywords = hit["_source"]["table_keywords"]

        term_vectors = client.termvectors(
            index=index_name,
            id=doc_id,
            fields=["table_keywords"],
            term_statistics=True,
        )

        term_count = 0
        if "table_keywords" in term_vectors["term_vectors"]:
            for term in term_vectors["term_vectors"]["table_keywords"]["terms"]:
                if term in tokens:
                    term_count += term_vectors["term_vectors"]["table_keywords"]["terms"][term][
                        "term_freq"
                    ]
        text_results_dict[table_nm_kor] = (table_keywords, term_count)

    vector_scores_dict = {
        hit["_source"]["table_name_kor"]: hit["_score"]
        for hit in vector_search_results["hits"]["hits"]
    }

    combined_results = text_search_results.copy()
    for hit in combined_results["hits"]["hits"]:
        table_nm_kor = hit["_source"]["table_name_kor"]
        vector_scores = vector_scores_dict.get(table_nm_kor, 0)
        if table_nm_kor in text_results_dict:
            _, term_count = text_results_dict[table_nm_kor]
        else:
            term_count = 0
        combined_score = term_count + vector_scores

        hit["_source"]["term_counts"] = term_count
        hit["_source"]["vector_score"] = vector_scores
        hit["_source"]["combined_score"] = combined_score

    normalized_results = normalized_score(combined_results)
    final_results = results_with_threshold_score(normalized_results, threshold_score)

    print(f"---Hybrid Search Result Threshold Score:{threshold_score}---")
    print("---Hybrid Search Completed---")
    return final_results


def filter_by_categories(results, filter_word):
    if not filter_word:
        return results
    categories = {filter_word} if isinstance(filter_word, str) else set(filter_word)
    filtered_hits = [
        hit for hit in results["hits"]["hits"] if hit["_source"].get("category") in categories
    ]
    results["hits"]["hits"] = filtered_hits
    return results


def normalized_score(results):
    scores = [hit["_source"]["combined_score"] for hit in results["hits"]["hits"]]
    if not scores:
        return results
    max_score = max(scores)
    if max_score == 0:
        return results
    normalization_factor = 100 / max_score
    for hit in results["hits"]["hits"]:
        s = hit["_source"]["combined_score"] * normalization_factor
        hit["_source"]["normalized_score"] = s
    sorted_results = sorted(
        results["hits"]["hits"],
        key=lambda x: x["_source"].get("normalized_score", 0),
        reverse=True,
    )
    results["hits"]["hits"] = sorted_results
    return results


def results_with_threshold_score(results, threshold_score):
    final_hits = []
    for hit in results["hits"]["hits"]:
        if hit["_source"].get("normalized_score", 0) >= threshold_score:
            final_hits.append(hit)
    results["hits"]["hits"] = final_hits
    return results
