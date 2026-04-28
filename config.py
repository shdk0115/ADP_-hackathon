# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# AWS CONFIGS
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_TEXT_MODEL = os.getenv("BEDROCK_TEXT_MODEL", "amazon.titan-text-lite-v1")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
--------------

### DEFAULT_INDEX
OPENSEARCH_DEFAULT = {
    "number_of_shards": 1,
    "number_of_replicas": 1,
    "knn": True
}

EMBEDDING_CONFIG_DEFAULT = {
    "dimension": 768,
    "similarity": "cosine"  # for OpenSearch 2.x/3.x
}

--------------
