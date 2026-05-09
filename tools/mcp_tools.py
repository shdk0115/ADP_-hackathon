"""
MCP-style tool registry.
Each tool has a Claude-compatible schema and a Python handler.
The data analyst agent calls these via Bedrock tool_use blocks.
"""
from typing import Any, Dict, List

from tools.keyword_extraction import extract_keywords
from tools.hyde import build_hyde_content, summarize_text
from tools.add_dcr import get_decompound_rules, expand_keywords_with_dcr
from tools.analyzer import build_nori_analyzer_config

# ── Tool schemas (Anthropic tool_use format) ──────────────────────────────────
TOOL_SCHEMAS: List[Dict] = [
    {
        "name": "extract_keywords",
        "description": "Extract significant keywords from text using LLM.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "summarize_text",
        "description": "Summarize text concisely in the same language.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "build_hyde_content",
        "description": "Build HyDE-enriched content: TEXT | summary | matched QA questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "qa_sheet": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["text", "qa_sheet"],
        },
    },
    {
        "name": "get_decompound_rules",
        "description": "Build Nori DCR user_dictionary_rules from keywords via OpenSearch analyze API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "index_name": {"type": "string"},
                "decompound_mode": {"type": "string", "default": "mixed"},
            },
            "required": ["keywords", "index_name"],
        },
    },
    {
        "name": "expand_keywords_with_dcr",
        "description": "Expand keywords using the nori_custom DCR analyzer on a target index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "index_name": {"type": "string"},
            },
            "required": ["keywords", "index_name"],
        },
    },
    {
        "name": "build_nori_analyzer_config",
        "description": "Build OpenSearch nori analyzer settings dict for index creation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decompound_mode": {"type": "string", "default": "mixed"},
                "dcr_rules": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
    },
]


def dispatch_tool(name: str, inputs: Dict[str, Any], os_client=None, qa_sheet=None) -> Any:
    """Execute a tool by name with the given inputs. Returns serialisable result.

    qa_sheet (when provided by the caller) overrides any qa_sheet the LLM may have
    fabricated in `inputs`, since the LLM does not know the real QA records.
    """
    if name == "extract_keywords":
        return extract_keywords(inputs["text"])

    if name == "summarize_text":
        return summarize_text(inputs["text"])

    if name == "build_hyde_content":
        qa = qa_sheet if qa_sheet is not None else inputs.get("qa_sheet", [])
        return build_hyde_content(inputs["text"], qa)

    if name == "get_decompound_rules":
        if os_client is None:
            return []
        return get_decompound_rules(
            inputs["keywords"], os_client, inputs["index_name"],
            inputs.get("decompound_mode", "mixed"),
        )

    if name == "expand_keywords_with_dcr":
        if os_client is None:
            return inputs["keywords"]
        return expand_keywords_with_dcr(inputs["keywords"], os_client, inputs["index_name"])

    if name == "build_nori_analyzer_config":
        return build_nori_analyzer_config(
            inputs.get("decompound_mode", "mixed"),
            inputs.get("dcr_rules", []),
        )

    raise ValueError(f"Unknown tool: {name}")
