from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import json
import re
import unicodedata
from tools.hyde import build_hyde_content
from tools.add_dcr import get_decompound_rules, expand_keywords_with_dcr
from tools.file_reader import read_raw_data


@dataclass
class ChunkDocument:
    id: str
    meta_data: Dict
    contents: str
    keywords: List[str]
    embedding: List[float]


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_korean_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _nori_like_analyze_tokens(text: str) -> List[str]:
    """
    Lightweight nori-like tokenization for local preprocessing.
    In production, replace with OpenSearch _analyze API results.
    """
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
    return [tok for tok in tokens if tok]


def _build_dcr_terms(text: str, llm_keywords: List[str]) -> List[str]:
    """
    DCR generation flow:
    1) llm keyword extraction terms
    2) nori analyzer terms from input text
    3) merge and augment spacing variants
    """
    analyzer_tokens = _nori_like_analyze_tokens(text)
    base_terms = llm_keywords + analyzer_tokens

    merged: List[str] = []
    seen = set()
    for term in base_terms:
        term = term.strip().lower()
        if not term:
            continue
        # original
        if term not in seen:
            seen.add(term)
            merged.append(term)
        # no-space variant
        no_space = term.replace(" ", "")
        if no_space and no_space not in seen:
            seen.add(no_space)
            merged.append(no_space)
        # simple bi-gram spacing variant (for Korean spacing confusion)
        if len(no_space) >= 4:
            mid = len(no_space) // 2
            spaced = f"{no_space[:mid]} {no_space[mid:]}"
            if spaced not in seen:
                seen.add(spaced)
                merged.append(spaced)

    return merged


def parse_retail_stock_rows(text: str) -> List[Dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows: List[Dict] = []
    current_category = ""
    current: Dict = {}

    def _flush():
        nonlocal current
        if current.get("product"):
            rows.append(current)
        current = {}

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            _flush()
            current_category = line.strip("[]")
            continue
        if line.startswith("상품명:"):
            _flush()
            current = {"category": current_category, "product": line.split(":", 1)[1].strip()}
        elif line.startswith("가격:"):
            current["price"] = line.split(":", 1)[1].strip()
        elif line.startswith("평점:"):
            current["rating"] = line.split(":", 1)[1].strip()
        elif line.startswith("해시태그:"):
            current["hashtags"] = line.split(":", 1)[1].strip()
        elif line.startswith("추천상황:"):
            current["use_case"] = line.split(":", 1)[1].strip()
    _flush()
    return rows


def build_concat_content(row: Dict) -> str:
    parts = [
        f"카테고리 {row.get('category', '')}",
        f"상품명 {row.get('product', '')}",
        f"가격 {row.get('price', '')}",
        f"평점 {row.get('rating', '')}",
        f"해시태그 {row.get('hashtags', '')}",
        f"추천상황 {row.get('use_case', '')}",
    ]
    return _normalize_text(" | ".join(parts))


def chunk_text_default_v1(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    sentences = _split_korean_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)

    # Character-level overlap for retrieval stability.
    if overlap > 0 and len(chunks) > 1:
        overlap_chunks: List[str] = [chunks[0]]
        for idx in range(1, len(chunks)):
            prev_tail = chunks[idx - 1][-overlap:]
            overlap_chunks.append(f"{prev_tail} {chunks[idx]}".strip())
        return overlap_chunks
    return chunks


def _read_one(file_path: Path) -> str:
    try:
        text = read_raw_data(str(file_path), filename=file_path.name)
    except Exception:
        return ""
    return (text or "").strip()


def load_raw_documents(raw_path: str) -> List[Dict]:
    """Read TXT / CSV / XLSX / PDF (single file or directory) into doc records."""
    path = Path(raw_path)
    if not path.exists():
        return []

    if path.is_file():
        text = _read_one(path)
        if not text:
            return []
        return [{"source": path.name, "text": text, "meta_data": {"source": path.name}}]

    docs: List[Dict] = []
    for file_path in sorted(path.glob("*")):
        if not file_path.is_file():
            continue
        text = _read_one(file_path)
        if not text:
            continue
        docs.append(
            {
                "source": file_path.name,
                "text": text,
                "meta_data": {"source": file_path.name},
            }
        )
    return docs


from tools.embedding import embed as bedrock_embed  # cached Titan v2 client


def preprocess_default(
    raw_docs: List[Dict],
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict]:
    embed = embedding_fn or bedrock_embed
    records: List[Dict] = []
    for doc in raw_docs:
        parsed_rows = parse_retail_stock_rows(doc["text"])
        if parsed_rows:
            for i, row in enumerate(parsed_rows):
                content = build_concat_content(row)
                records.append(
                    {
                        "id": f"{doc['source']}-row-{i}",
                        "meta_info": {
                            **doc.get("meta_data", {}),
                            "category": row.get("category", ""),
                            "product": row.get("product", ""),
                            "price": row.get("price", ""),
                            "rating": row.get("rating", ""),
                        },
                        "content": content,
                        "embedding": embed(content),
                    }
                )
            continue

        chunks = chunk_text_default_v1(doc["text"], chunk_size=500, overlap=0)
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{doc['source']}-{i}",
                    "meta_info": doc.get("meta_data", {}),
                    "content": chunk,
                    "embedding": embed(chunk),
                }
            )
    return records


def preprocess_agentic(
    raw_docs: List[Dict],
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
    keyword_fn: Optional[Callable[[str], List[str]]] = None,
    use_analyzer_dcr: bool = False,
    use_hyde: bool = False,
    qa_sheet: Optional[List[Dict]] = None,
    os_client=None,
    dcr_index: str = "_dcr_temp",
) -> List[Dict]:
    """
    Agentic preprocessing for v2/v3:
    - v2: keyword extraction + keyword column
    - v3: keywords → nori DCR rules → concat keywords + analyzed tokens (deduped)
         + HyDE summary appended to content
    """
    embed = embedding_fn or bedrock_embed
    records: List[Dict] = []

    for doc in raw_docs:
        parsed_rows = parse_retail_stock_rows(doc["text"])

        if parsed_rows:
            # Retail-style structured rows
            for i, row in enumerate(parsed_rows):
                content = build_concat_content(row)
                hyde_text = ""
                if use_hyde:
                    content = build_hyde_content(content, qa_sheet or [])

                if keyword_fn is not None:
                    keywords = keyword_fn(content)
                else:
                    tags = row.get("hashtags", "")
                    keywords = [tok.strip("#").lower() for tok in tags.split() if tok.startswith("#")]

                dcr_terms: List[str] = []
                if use_analyzer_dcr:
                    if os_client is not None:
                        dcr_terms = expand_keywords_with_dcr(keywords, os_client, dcr_index)
                    else:
                        dcr_terms = _build_dcr_terms(content, keywords)
                    keywords = dcr_terms

                records.append(
                    {
                        "id": f"{doc['source']}-agent-{i}",
                        "metainfo": {
                            **doc.get("meta_data", {}),
                            "category": row.get("category", ""),
                            "product": row.get("product", ""),
                            "price": row.get("price", ""),
                            "rating": row.get("rating", ""),
                        },
                        "TEXT": content,
                        "keyword": " ".join(keywords),
                        "embedding": embed(content),
                    }
                )
            continue

        # Fallback: free-text / PDF / Excel-as-text → sentence chunking
        chunks = chunk_text_default_v1(doc["text"], chunk_size=500, overlap=0)
        for i, chunk in enumerate(chunks):
            content = chunk
            if use_hyde:
                content = build_hyde_content(content, qa_sheet or [])

            keywords = keyword_fn(content) if keyword_fn is not None else []

            dcr_terms = []
            if use_analyzer_dcr:
                if os_client is not None:
                    dcr_terms = expand_keywords_with_dcr(keywords, os_client, dcr_index)
                else:
                    dcr_terms = _build_dcr_terms(content, keywords)
                keywords = dcr_terms

            records.append(
                {
                    "id": f"{doc['source']}-agent-{i}",
                    "metainfo": doc.get("meta_data", {}),
                    "TEXT": content,
                    "keyword": " ".join(keywords),
                    "embedding": embed(content),
                }
            )

    return records


def save_processed_json(records: List[Dict], output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
