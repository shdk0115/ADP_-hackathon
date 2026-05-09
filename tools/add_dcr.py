from typing import List


def get_decompound_rules(
    keywords: List[str],
    os_client,
    index_name: str,
    decompound_mode: str = "mixed",
) -> List[str]:
    """
    Build DCR user_dictionary_rules:
    1. keywords → analyze with DEFAULT nori tokenizer (no DCR yet)
    2. dedupe original keywords + nori tokens → final rule list
    """
    seen = set()
    rules: List[str] = []

    def _add(term: str):
        t = term.strip().lower()
        if t and t not in seen:
            seen.add(t)
            rules.append(t)

    for kw in keywords:
        _add(kw)

    all_text = " ".join(keywords)
    if all_text.strip():
        try:
            response = os_client.indices.analyze(
                body={
                    "tokenizer": {"type": "nori_tokenizer", "decompound_mode": decompound_mode},
                    "text": all_text,
                },
            )
            for token_obj in response.get("tokens", []):
                _add(token_obj["token"])
        except Exception:
            pass

    return rules


def expand_keywords_with_dcr(
    keywords: List[str],
    os_client,
    index_name: str,
) -> List[str]:
    """
    Expand keywords for ingestion:
    1. keywords → analyze with DCR analyzer (nori_custom on the target index)
    2. concat original + analyzed tokens, NO dedupe
    """
    expanded = list(keywords)

    all_text = " ".join(keywords)
    if all_text.strip():
        try:
            response = os_client.indices.analyze(
                index=index_name,
                body={"analyzer": "nori_custom", "text": all_text},
            )
            expanded += [t["token"] for t in response.get("tokens", [])]
        except Exception:
            pass

    return expanded
