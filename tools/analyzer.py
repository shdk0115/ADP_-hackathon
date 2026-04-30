from typing import Dict


def build_nori_analyzer_config(decompound_mode: str = "mixed") -> Dict:
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
                "nori_tokenizer": {
                    "type": "nori_tokenizer",
                    "decompound_mode": decompound_mode,
                }
            },
        }
    }
