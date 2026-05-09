from typing import Dict, List


def build_nori_analyzer_config(decompound_mode: str = "mixed", dcr_rules: List[str] = None) -> Dict:
    tokenizer_config = {
        "type": "nori_tokenizer",
        "decompound_mode": decompound_mode,
    }
    if dcr_rules:
        tokenizer_config["user_dictionary_rules"] = dcr_rules

    return {
        "analysis": {
            "analyzer": {
                "nori_custom": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["lowercase"],
                }
            },
            "tokenizer": {
                "nori_tokenizer": tokenizer_config,
            },
        }
    }
