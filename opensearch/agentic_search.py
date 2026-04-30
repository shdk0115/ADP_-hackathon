
def select_custom_hybrid_query_w_score_parallel(client, question, index_name, query_size, filter_word, threshold_score=0, text_min_score=1):
    '''
    Args:
        client : Opensearch Client
        question : Input Question 
        index_name : Opensearch Index Name
        query_size : query_size (k size) -- OpenSearch에서 조회 하고 싶은 최대 문서 개수
        filter_word : Filter 하고 싶은 단어
        threshold_score : 정규화된 유사도 점수 기준 적용할 Threshold Score (Ex. TS =70, 70점 이상인 문서들만 조회)
        text_min_score : 1, "100%" -- 변경 가능
        
    Return:
        results : Opensearch 조회 결과
            VERSION 1 - Term Count 0 미포함
            VERSION 2 - Term Count 0 포함
    '''
    # Text Search를 위해 질문을 Decompound Rule 적용된 분석기로 쪼개기 및 TERM COUNT 위한 토큰 출력
    analyzer = "nori_analyzer_w_dcr"
    analyzed_question, tokens = nori_analyze_w_token(client, index_name, analyzer, question)
    print("Text Search Input Question:", question)
    print("Hybrid Search Input Question:", analyzed_question)
    # Vector Search를 위해 질문 임베딩 적용
    region = 'us-east-1'
    embedding_model_id = 'amazon.titan-embed-text-v1'
    embedded_query = text_embedding_with_bedrock(question, region, embedding_model_id)
    
    # TEXT와 VECTOR SEARCH 병렬로 조회
    with ThreadPoolExecutor() as executor:
        # TEXT SEARCH
        text_search_future = executor.submit(text_search, client, analyzed_question, index_name, query_size, filter_word, text_min_score)
        # VECTOR SEARCH
        vector_search_future = executor.submit(vector_search, client, embedded_query, index_name, query_size)
        
        # 조회 결과들
        text_search_results = text_search_future.result()
        vector_search_results = vector_search_future.result()
        # print(vector_search_results)
    
    # TEXT SEARCH 결과들 중 TERM COUNT 지행
    text_results_dict = {}
    for hit in text_search_results['hits']['hits']:
        doc_id = hit['_id']
        table_nm_kor = hit['_source']['table_name_kor']
        table_keywords = hit['_source']['table_keywords']

        term_vectors = client.termvectors(index= index_name, id= doc_id, fields=['table_keywords'], term_statistics= True)

        term_count = 0
        if 'table_keywords' in term_vectors['term_vectors']:
            for term in term_vectors['term_vectors']['table_keywords']['terms']:
                if term in tokens:
                    term_count += term_vectors['term_vectors']['table_keywords']['terms'][term]['term_freq']
        # 특정 테이블 내에 어떤 키워드들이 몇번 매칭 됬는지 결과 저장 (Ex. 수익성분석 = ('매출',2))
        text_results_dict[table_nm_kor] = (table_keywords, term_count)
    
    # SAVE VECTOR SEARCH RESULTS
    vector_scores_dict = {hit['_source']['table_name_kor']: hit['_score'] for hit in vector_search_results['hits']['hits']}

    # VERSION 1 vs. VERSION 2중 택일 (다른건 주석처리)
    ########################################
    # VERSION 1 
    ##########
    combined_results = text_search_results.copy() # Based on TEXT SEARCH RESULTS
    ##########
    
    # # VERSION 2
    # ##########
    # FILTER VECTOR RESULTS
    # vector_search_results = filter_by_categories(vector_search_results, filter_word)
    # # COMBINED RESULTS
    # combined_results = vector_search_results.copy() # Based on VECTOR SEARCH RESULTS
    ##########
    ########################################
    # CALCULATE COMBINED SCORE
    for hit in combined_results['hits']['hits']:
        table_nm_kor = hit['_source']['table_name_kor']
        vector_scores = vector_scores_dict.get(table_nm_kor, 0)

        if table_nm_kor in text_results_dict:
            _, term_count = text_results_dict[table_nm_kor]
        else:
            term_count = 0
        combined_score = term_count + vector_scores

        hit['_source']['term_counts'] = term_count
        hit['_source']['vector_score'] = vector_scores
        hit['_source']['combined_score'] = combined_score

    # NORMALIZE and APPLY THRSHOLD SCORE
    normalized_results = normalized_score(combined_results)
    final_results = results_with_threshold_score(normalized_results, threshold_score)

    print(f"---Hybrid Search Result Threshold Score:{threshold_score}---")
    print(f"---Hybrid Search Completed---")
    return final_results

# select_custom_hybrid_query_w_score_parallel VERSION2에 필요한 필터링 기능
def filter_by_categories(results, filter_word):
    '''
    Args:
        results : OpenSearch 조회 결과
        filter_word : Filter 하고 싶은 단어
        
    Return:
        filtered_results : Filter_Word로 필터된 Opensearch 조회 결과
    '''
    if not filter_word: 
        return results
    
    filtered_hits = [hit for hit in results['hits']['hits'] if hit['_source']['category'] in categories]
    results['hits']['hits'] = filtered_hits
    return results

# 유사도 점수 정규화 - 1등 값을 100으로 간주하고 나머지 치환
def normalized_score(results):
    '''
    Args:
        results : OpenSearch 조회 결과
        
    Return:
        normalized_results : 정규화된 Opensearch 조회 결과
    '''
    scores= [hit['_source']['combined_score'] for hit in results['hits']['hits']]
    max_score = max(scores)
    normalization_factor = 100 / max_score

    for hit in results['hits']['hits']:
        normalized_score = hit['_source']['combined_score'] * normalization_factor
        hit['_source']['normalized_score'] = normalized_score
    sorted_results = sorted(results['hits']['hits'], key = lambda x: x['_source']['normalized_score'], reverse=True)
    results['hits']['hits'] = sorted_results
    return results

# 정규화 된 점수 기준 
def results_with_threshold_score(results, threshold_score):
    '''
    Args:
        results : OpenSearch 조회 결과
        threshold_score : 정규화된 유사도 점수 기준 적용할 Threshold Score (Ex. TS =70, 70점 이상인 문서들만 조회)
        
    Return:
        results above threshold_score : 유사도 점수가 Threshold Score 보다 높은 OpenSearch 조회 결과
    '''
    final_hits = [] 
    for hit in results['hits']['hits']:
        if hit['_source']['normalized_score'] >= threshold_score:
            final_hits.append(hit)
    results['hits']['hits'] = final_hits
    return results
