STRATEGY_DECISION_PROMPT = """\
You are an OpenSearch indexing strategy architect.
Based on the data profile below, decide the optimal indexing strategy.
You MUST follow the decision rules exactly. Do not deviate or use your own judgment.

## Decision Rules (apply in order)

### use_analyzer
- true  if language.ko_ratio >= 0.5
- false otherwise

### use_keyword
- true  if content.keyword_density is "medium" or "high"
- true  if domain.specificity is "high"
- true  if language.has_technical_terms is true
- false otherwise (all conditions must be false)

### use_hyde
- true  if content.semantic_complexity is "high"
- true  if domain.type is "legal", "medical", or "finance"
- false otherwise

### use_hybrid
- true  if use_keyword is true
- false otherwise

### use_dcr
- true  if use_analyzer is true AND quality.noise_level is "high"
- false otherwise

### index_version
- "v3" if use_analyzer is true AND use_hyde is true
- "v2" if use_keyword is true (and not v3)
- "v1" otherwise

### fields
- Always include: meta_data (object), contents (text), embedding (knn_vector)
- Add keywords (text) only if use_keyword is true

### analyzer_config
- If use_analyzer is true:
    type: "nori"
    decompound_mode: "mixed"  if domain.specificity is "high"
    decompound_mode: "discard" otherwise
- If use_analyzer is false: null

### chunking_strategy
- "structure-aware" if structure.has_consistent_schema is true
- "sentence"        if content.length_variance is "high"
- "fixed"           otherwise

## Strict Output Rules
- Return ONLY a valid JSON object.
- No markdown, no explanation, no preamble.
- use_vector must always be true.
- decision_log must list every rule that fired, written as a human-readable string.
  Example: "ko_ratio 0.85 >= 0.5 → use_analyzer: true (nori)"

## Output Schema
{{
  "index_version": "v1" | "v2" | "v3",
  "use_keyword": bool,
  "use_vector": true,
  "use_hybrid": bool,
  "use_analyzer": bool,
  "use_hyde": bool,
  "use_dcr": bool,
  "fields": {{
    "meta_data": "object",
    "contents": "text",
    "embedding": "knn_vector"
  }},
  "analyzer_config": {{ "type": string, "decompound_mode": string }} | null,
  "chunking_strategy": "fixed" | "sentence" | "structure-aware",
  "decision_log": [string]
}}

## Data Profile
{data_profile}
"""


RETRY_PROMPT = """\
Your previous response failed validation.

## Validation Error
{error_message}

## Your Previous Response
{previous_response}

Fix ONLY the validation error and return valid JSON only.
No markdown, no explanation, no preamble.
"""
