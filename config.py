import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

_CONFIG_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(_CONFIG_DIR / ".env")
    load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "us.anthropic.claude-sonnet-4-6")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

OPENSEARCH_DEFAULTS = {
    "number_of_shards": 1,
    "number_of_replicas": 1,
    "knn": True,
}

EMBEDDING_CONFIG = {
    "dimension": 1024,
    "similarity": "cosine",
}

DEFAULT_INDEX_NAME = "docs_default_v1"
AGENT_INDEX_PREFIX = "docs_agent"
MAX_ANALYSIS_RETRY = int(os.getenv("MAX_ANALYSIS_RETRY", "3"))

# ── 파이프라인 전략 retry ───────────────────────────
# V1 → V2 → V3 ... 반복 최대 횟수
MAX_PIPELINE_RETRY = int(os.getenv("MAX_PIPELINE_RETRY", "3"))

# ── 목표 정확도 ─────────────────────────────────────
# 이 점수 이상이면 파이프라인 조기 종료
# 0~100 사이 정수, .env 또는 여기서 직접 변경 가능
TARGET_SCORE = int(os.getenv("TARGET_SCORE", "95"))
SEARCH_K = int(os.getenv("SEARCH_K", "3"))
