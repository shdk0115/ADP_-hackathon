"""tools/file_reader.py — Raw Data(TXT/Excel/PDF) 및 QA Sheet(Excel/PDF) 읽기"""
from __future__ import annotations
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Union

_QA_COL_MAP = {
    "질문": "question", "question": "question", "user question": "question",
    "기대 정답": "expected_answer", "기대정답": "expected_answer",
    "expected_answer": "expected_answer", "expected answer": "expected_answer",
    "정답": "expected_answer", "answer": "expected_answer",
    "난이도": "difficulty", "difficulty": "difficulty",
    "id": "id", "번호": "id", "no": "id", "no.": "id",
    "체크포인트": "checkpoint", "체크 포인트": "checkpoint",
    "checkpoint": "checkpoint",
    "에이전트 체크 포인트": "checkpoint", "에이전트 체크포인트": "checkpoint",
    "agent checkpoint": "checkpoint",
}


def _normalize_col(col: str) -> str:
    """Strip '(보조설명)' parentheticals and squeeze whitespace, then lowercase."""
    s = str(col)
    s = re.sub(r"\s*[\(（][^\)）]*[\)）]\s*", " ", s)  # remove (...) and （...）
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def read_raw_data(source, filename="") -> str:
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = path.name
        with open(path, "rb") as f:
            source = f.read()
    elif hasattr(source, "read"):
        source = source.read()

    ext = Path(filename).suffix.lower()
    if ext == ".txt":           return _read_txt(source)
    elif ext in (".xlsx",".xls",".csv"): return _read_excel_as_text(source, ext)
    elif ext == ".pdf":         return _read_pdf(source)
    else:
        for enc in ("utf-8","cp949","euc-kr"):
            try: return source.decode(enc)
            except: continue
        return source.decode("utf-8", errors="replace")

def _read_txt(data):
    for enc in ("utf-8","cp949","euc-kr"):
        try: return data.decode(enc)
        except: continue
    return data.decode("utf-8", errors="replace")

def _read_excel_as_text(data, ext):
    try:
        import pandas as pd
        buf = io.BytesIO(data)
        if ext == ".csv":
            df = None
            for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                return "[Excel 읽기 실패: CSV 인코딩 미상 (utf-8/cp949/euc-kr 모두 실패)]"
        else:
            df = pd.read_excel(buf, engine="openpyxl")
        return df.to_string(index=False)
    except Exception as e:
        return f"[Excel 읽기 실패: {e}]"

def _read_pdf(data):
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        return "[PDF 읽기 실패: pip install pypdf]"
    except Exception as e:
        return f"[PDF 읽기 실패: {e}]"

def read_qa_sheet_excel(source, filename="") -> List[Dict]:
    import pandas as pd
    if isinstance(source, (str, Path)):
        filename = Path(source).name
        with open(source, "rb") as f: data = f.read()
    elif hasattr(source, "read"): data = source.read()
    else: data = source

    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(data)
    if ext == ".csv":
        df = None
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                buf.seek(0)
                df = pd.read_csv(buf, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"CSV 디코딩 실패 (utf-8/cp949/euc-kr 모두 실패): {filename}")
    else:
        df = pd.read_excel(buf, engine="openpyxl")
    # Normalize: strip "(...)" hints, lowercase, then map to canonical names
    rename_map = {}
    for c in df.columns:
        norm = _normalize_col(c)
        if norm in _QA_COL_MAP:
            rename_map[c] = _QA_COL_MAP[norm]
    df = df.rename(columns=rename_map)

    if "question" not in df.columns:
        raise ValueError(f"'질문' 컬럼 없음. 현재: {list(df.columns)}")
    if "expected_answer" not in df.columns:
        raise ValueError(f"'기대 정답' 컬럼 없음. 현재: {list(df.columns)}")

    records = []
    for i, row in df.iterrows():
        rec = {"id": str(row.get("id", f"Q{i+1}")),
               "question": str(row["question"]).strip(),
               "expected_answer": str(row["expected_answer"]).strip()}
        for opt in ("difficulty", "checkpoint"):
            if opt in df.columns:
                rec[opt] = str(row.get(opt,"")).strip()
        records.append(rec)
    return records


# ──────────────────────────────────────────────────────
# PDF QA: extract (question, expected_answer) pairs via LLM
# ──────────────────────────────────────────────────────
_QA_EXTRACTION_PROMPT = """\
다음 문서에서 평가용 QA(질문/기대 정답) 쌍을 추출하세요.
- 문서 자체가 QA 시트일 수도 있고, 일반 본문일 수도 있습니다.
- QA 시트면 표/리스트 형태의 질문-정답 쌍을 그대로 추출하세요.
- 일반 본문이면 핵심 사실에 대한 질문과 정답을 도출하세요.
- 정답이 문서 안에 명확히 존재하는 항목만 포함하세요.
- 최대 30개.

## 출력 규칙
- JSON array 만 출력. 다른 설명/마크다운/코드펜스 금지.
- 각 항목 스키마: {{"id":"Q1","question":"...","expected_answer":"..."}}

## 문서
{document}
"""


def _llm_extract_qa(text: str, max_chars: int = 12000) -> List[Dict]:
    """Call Bedrock Claude to extract QA pairs from raw text."""
    try:
        import boto3
        from config import AWS_REGION, BEDROCK_LLM_MODEL
    except ImportError as e:
        raise RuntimeError(f"PDF QA 추출에는 boto3/config 필요: {e}")

    snippet = text[:max_chars]
    prompt = _QA_EXTRACTION_PROMPT.format(document=snippet)

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(
        modelId=BEDROCK_LLM_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    raw = json.loads(response["body"].read())["content"][0]["text"].strip()

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []

    out: List[Dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("expected_answer") or item.get("answer") or "").strip()
        if not q or not a:
            continue
        out.append({
            "id": str(item.get("id", f"Q{i+1}")),
            "question": q,
            "expected_answer": a,
        })
    return out


def read_qa_sheet_pdf(source, filename="") -> List[Dict]:
    """Extract QA pairs from a PDF using an LLM."""
    text = read_raw_data(source, filename=filename)
    if not text or not text.strip():
        raise ValueError(f"PDF 텍스트가 비어 있음: {filename or source}")
    if text.startswith("[PDF 읽기 실패"):
        raise ValueError(text)
    pairs = _llm_extract_qa(text)
    if not pairs:
        raise ValueError(f"PDF에서 QA 쌍을 추출하지 못함: {filename or source}")
    return pairs