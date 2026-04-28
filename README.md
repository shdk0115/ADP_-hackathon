import os

# Define the content of the markdown file
md_content = """# ADP Hackathon Project Structure

This repository contains the source code and data for the ADP Hackathon project, focusing on agentic search and retrieval using AWS Bedrock and OpenSearch.

## Directory Layout

```text
ADP_-hackathon/
├── .env                    # Environment variables (API keys, endpoints)
├── config.py               # Global configuration and constants
├── main.py                 # Application entry point
│
├── agents/                 # Orchestration logic for autonomous agents
│   └── opensearch_agent.py      # Agent responsible for indexing decisions
│   └── data_analysis.py    # Agent responsible for analyzing income data
│
├── clients/                # Third-party service clients
│   └── bedrock_client.py   # AWS Bedrock interface for LLM/Embeddings
│
├── data/                   # Data storage
│   ├── qa_sheet/           # Evaluation datasets (Questions/Answers)
│   └── raw_data/           # Source documents before processing
│
├── evaluation/             # Testing and performance metrics
│   ├── error_analysis/     # Logs and reports on retrieval failures
│   └── evaluator.py        # Script to run RAG evaluation (e.g., Ragas/DeepEval)
│
├── tools/                  # Utility functions and RAG enhancement techniques
│   ├── keyword_extraction.py # NLP tools to pull keywords from queries
│   ├── hyde.py             # Hypothetical Document Embeddings implementation
│   ├── analyzer.py         # Text analysis and statistics
│   ├── add_dcr.py          # Document Content Re-ranking/processing
│   └── preprocessor.py     # Data cleaning and chunking logic
│
└── opensearch/             # OpenSearch integration and search logic
    ├── default_index.py    # Standard indexing scripts
    ├── default_search.py   # Baseline search implementation
    ├── agentic_index.py    # Advanced indexing for agent-based retrieval
    ├── agentic_search.py   # Multi-step or tool-augmented search logic
    └── ingest.py           # Data ingestion pipeline
