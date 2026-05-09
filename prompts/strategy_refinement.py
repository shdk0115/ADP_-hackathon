STRATEGY_REFINEMENT_PROMPT = """\
You are an OpenSearch indexing strategy optimizer.
The current pipeline failed to reach the target score.
Analyze the error report and refine the strategy to fix the failures.

## Current Strategy (what was tried)
{current_strategy}

## Error Report (what went wrong)
{error_report}

## Error Pattern Guide
Use this guide to decide what to change:

CONDITIONAL_REASONING_FAIL
  → The search returned the base rule but missed the exception clause.
  → Fix: increase use_hyde, switch chunking_strategy to "sentence"

ENTITY_MAPPING_FAIL
  → A renamed entity (e.g. "AI 유닛" → "AIR 유닛") was not recognized.
  → Fix: set use_keyword true, add synonym handling hint to decision_log

DATE_BOUNDARY_FAIL
  → A date-based condition was not correctly applied.
  → Fix: use_hyde true, note date reasoning needed in decision_log

TABLE_RETRIEVAL_FAIL
  → Structured table data was not retrieved correctly.
  → Fix: use_keyword true, chunking_strategy "structure-aware"

ANALYZER_ERROR
  → Morphological analysis caused wrong tokenization.
  → Fix: change analyzer_config.decompound_mode from "mixed" to "discard" or vice versa

VECTOR_ONLY_INSUFFICIENT
  → Semantic search alone could not match domain-specific terms.
  → Fix: use_keyword true, use_hybrid true

## Strict Output Rules
- Return ONLY a valid JSON object. No markdown, no explanation.
- You may only UPGRADE settings, never downgrade index_version.
- use_vector must always be true.
- Update decision_log to explain every change made.
- Preserve all fields from current strategy that are not being changed.

## Output Schema (same as current strategy)
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
    "keywords": "text",
    "embedding": "knn_vector"
  }},
  "analyzer_config": {{ "type": string, "decompound_mode": string }} | null,
  "chunking_strategy": "fixed" | "sentence" | "structure-aware",
  "decision_log": [string]
}}
"""
