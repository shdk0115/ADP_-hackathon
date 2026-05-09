"""tools/file_reader.py — Raw Data(TXT/Excel/PDF) 및 QA Sheet(Excel) 읽기"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, List, Union

_QA_COL_MAP = {
    "질문": "question", "question": "question",
    "기대 정답": "expected_answer", "기대정답": "expected_answer",
    "expected_answer": "expected_answer", "정답": "expected_answer",
    "난이도": "difficulty", "difficulty": "difficulty",
    "id": "id", "ID": "id", "번호": "id",
    "체크포인트": "checkpoint", "checkpoint": "checkpoint",
}

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
        df = pd.read_csv(buf, encoding="utf-8") if ext==".csv" \
             else pd.read_excel(buf, engine="openpyxl")
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
    df = pd.read_csv(buf) if ext==".csv" else pd.read_excel(buf, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: _QA_COL_MAP[c] for c in df.columns if c in _QA_COL_MAP})

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