# ADP Hackathon Pipeline

This project runs an iterative OpenSearch indexing/search pipeline:

1. Build baseline (`v1`) without data-analysis-based tuning.
2. Build agent index (`v2`/`v3`) using Data Analysis Result.
3. Evaluate each step with QA sheet and run error analysis.

## Target Flow

### Iteration 1 (Baseline)
- Input: `user prompt + Data analysis result + QA sheet`
- Use **default index v1** only (ignore analysis result in this step)
- Preprocess -> ingest -> default search -> evaluation
- Output: baseline score

### Iteration 2 (Agent Index)
- Use Data Analysis Result to choose index version and options
- Preprocess -> create agent index -> ingest -> agent search -> evaluation
- Output: improved score + error analysis

### Iteration 3 (Retry)
- Adjust config from previous error analysis
  - Example: `v2` without analyzer/decompound mode
- Re-index -> search -> evaluate
- Output: best score among baseline/try2/try3

## Data Analysis Result Schema

```json
{
  "index_version": "v3",
  "use_keyword": true,
  "use_vector": true,
  "use_hybrid": true,
  "use_analyzer": true,
  "use_dcr": false,
  "fields": {
    "meta_data": "object",
    "contents": "text",
    "keywords": "text",
    "embedding": "knn_vector"
  },
  "analyzer_config": {
    "type": "nori",
    "decompound_mode": "mixed"
  }
}
```

## Index Versions

- `v1`: `meta_data`, `contents`, `embedding`
- `v2`: `meta_data`, `contents`, `keywords`, `embedding`
- `v3`: `v2` + analyzer + HyDE option (+ optional DCR)

## Team Deliverables

- Daniel
  - Data bucket: easy with `v1`, needs `v3`, fails even with `v3`
  - QA Sheet
- Seonghyeon
  - Data analysis prompt and criteria
- Paul
  - Default preprocessing code
  - Default search code (`v1`)
  - OpenSearch Agent (MCP tools)

## OpenSearch Agent Tool Hierarchy

- Keyword Extraction
- HyDE
- Analyzer
- DCR
- Ingest/Delete
- Search
- Data preprocessing

## Run (Local Simulation)

```bash
python main.py
```

`main.py` currently provides a local simulation for baseline/agent evaluation flow.
For production OpenSearch execution, connect your OpenSearch client and call:
- `opensearch/default_index.py`
- `opensearch/agentic_index.py`
- `opensearch/ingest.py`
- `opensearch/default_search.py`
- `opensearch/agentic_search.py`
