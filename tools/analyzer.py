# 기본 Nori Analyzer (analyzer라는 index 필수)
def default_nori_analyze(client, text):
    results = client.indices.analyze(
        index = 'analyzer',
        body = {
            'analyzer' : 'nori_analyzer',
            'text' : text
        }                  
    )
    results=list(results.values())[0]
    
    tokenize_sentence = ""
    for r in results:
        tokenize_sentence += r['token']+" "
    return tokenize_sentence

# Custom Nori - Token 출력 (Index/Analzer이름 변경 가능 Ex. Index Name: "nori-test"/ Analyzer: "nori_analyzer_w_dcr")
def nori_analyze_w_token(client, index_name, analyzer, text):
    results = client.indices.analyze(
        index = index_name,
        body = {
            'analyzer' : analyzer,
            'text': text
        }
    )
    tokens = [token['token']for token in results['tokens']]
    results=list(results.values())[0]

    tokenize_sentence = ""
    for r in results:
        tokenize_sentence += r['token'] + " "
    return tokenize_sentence, tokens
