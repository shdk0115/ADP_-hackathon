"""Bedrock Titan embedding wrapper.

- Default model: amazon.titan-embed-text-v2:0 (1024 dim, multilingual).
- LRU cache so repeated chunks/queries don't re-bill Bedrock.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List

import boto3

from config import AWS_REGION, BEDROCK_EMBED_MODEL, EMBEDDING_CONFIG

_DIM = EMBEDDING_CONFIG.get("dimension", 1024)
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def _build_body(text: str) -> str:
    if "titan-embed-text-v2" in BEDROCK_EMBED_MODEL:
        return json.dumps({"inputText": text, "dimensions": _DIM, "normalize": True})
    if "titan-embed" in BEDROCK_EMBED_MODEL:
        # v1 has no dimensions/normalize options
        return json.dumps({"inputText": text})
    if "cohere.embed" in BEDROCK_EMBED_MODEL:
        return json.dumps({"texts": [text], "input_type": "search_document"})
    return json.dumps({"inputText": text})


def _parse_vector(payload: dict) -> List[float]:
    if "embedding" in payload:
        return payload["embedding"]
    if "embeddings" in payload and payload["embeddings"]:
        first = payload["embeddings"][0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
        if isinstance(first, list):
            return first
    return []


@lru_cache(maxsize=4096)
def _embed_cached(text: str) -> tuple:
    if not text or not text.strip():
        return tuple([0.0] * _DIM)
    body = _build_body(text)
    resp = _get_client().invoke_model(
        modelId=BEDROCK_EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    vec = _parse_vector(payload)
    if not vec:
        return tuple([0.0] * _DIM)
    return tuple(vec)


def embed(text: str) -> List[float]:
    """Public entry point — returns a fresh list (mutable consumer-safe)."""
    print(f"  🔢 Embed 호출 ({len(text)}자)")
    return list(_embed_cached(text))
