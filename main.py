import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.error_analysis import analyze_errors
from config import (
    OPENSEARCH_HOST,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_USE_SSL,
    SEARCH_K,
    TARGET_SCORE,
)
from evaluation.evaluator import evaluate_qa_sheet
from opensearch.agentic_index import create_agent_index
from opensearch.agentic_search import search_v2, search_v3
from opensearch.default_index import create_index_v1
from opensearch.default_search import default_hybrid_search
from opensearch.ingest import ingest_documents_v1, ingest_documents_v2, ingest_documents_v3
from prompts.domain.corporate import CORPORATE_DOMAIN_HINT, CORPORATE_FEW_SHOT
from tools.add_dcr import get_decompound_rules
from tools.file_reader import read_qa_sheet_excel, read_qa_sheet_pdf
from tools.keyword_extraction import extract_keywords
from tools.preprocessing import bedrock_embed, load_raw_documents, preprocess_agentic, preprocess_default

try:
    from opensearchpy import OpenSearch
except ImportError:
    OpenSearch = None

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_STRATEGY = {
    "use_keyword": False,
    "use_vector": True,
    "use_hybrid": False,
    "use_analyzer": False,
    "use_hyde": False,
    "use_dcr": False,
    "fields": {"meta_info": "object", "TEXT": "text", "embedding": "knn_vector"},
    "chunking_strategy": "fixed",
    "decision_log": ["Sequential fixed strategy."],
}


def _resolve_data_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    return str(p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve())


# ──────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────
def _load_qa_sheet(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls", ".csv"):
        return read_qa_sheet_excel(str(p), filename=p.name)
    if ext == ".pdf":
        print(f"  📄 PDF QA detected → LLM 추출 시작 ({p.name})")
        pairs = read_qa_sheet_pdf(str(p), filename=p.name)
        print(f"  📄 PDF QA 추출 완료: {len(pairs)}개")
        return pairs
    if ext == ".json":
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                raw = p.read_text(encoding=enc).strip()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"QA JSON 디코딩 실패 (utf-8/cp949/euc-kr 모두 실패): {p.name}")
        return json.loads(raw) if raw else []
    raise ValueError(
        f"지원하지 않는 QA 파일 포맷: '{ext}' ({p.name}). "
        f"엑셀(.xlsx/.xls), CSV, PDF, JSON만 지원합니다."
    )


def _get_client():
    if OpenSearch is None:
        raise RuntimeError("opensearchpy is not installed. Run: pip install opensearch-py")
    auth = (OPENSEARCH_USER, OPENSEARCH_PASSWORD) if OPENSEARCH_USER and OPENSEARCH_PASSWORD else None
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def _reset_index(client, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)


def _score(eval_result: Dict) -> float:
    """evaluator 결과 → 0~100점 변환"""
    return round(eval_result.get("accuracy", 0.0) * 100, 1)


def _sep(title: str = "", width: int = 60) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'━' * pad} {title} {'━' * pad}")
    else:
        print("─" * width)


def _print_decision_log(logs: List[str]) -> None:
    print("  [AI 전략 결정 근거]")
    for log in logs:
        print(f"  → {log}")


def _print_eval_row(version: str, score: float, eval_result: Dict) -> None:
    correct = eval_result.get("correct", 0)
    total   = eval_result.get("total", 0)
    status  = "🎉 목표 달성" if score >= TARGET_SCORE else f"미달 ({TARGET_SCORE - score:.1f}점 부족)"
    print(f"  [{version}] {score:.1f}점  ({correct}/{total}개 정답)  {status}")


# ──────────────────────────────────────────────────────
# 검색 함수
# ──────────────────────────────────────────────────────
def _search_v1(client, index_name, query, size=SEARCH_K):
    qv = bedrock_embed(query)
    return default_hybrid_search(client, index_name, query, qv, size=size)


def _search_agent(client, index_name, query, strategy, size=SEARCH_K):
    qv = bedrock_embed(query)
    version = strategy.get("index_version", "v2")
    if version == "v3":
        return search_v3(client, index_name, query, qv, size=size)
    return search_v2(client, index_name, query, qv, size=size)


# ──────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────
def run_pipeline(
    user_prompt: str,
    raw_data_path: str = "data/복지규정_demo.txt",
    qa_sheet_path: str = "data/qa_sheet_demo.json",
) -> Dict:

    raw_path = _resolve_data_path(raw_data_path)
    qa_path  = _resolve_data_path(qa_sheet_path)

    raw_docs = load_raw_documents(raw_path)
    qa_sheet = _load_qa_sheet(qa_path)
    client   = _get_client()

    _sep("ADP 해커톤 파이프라인 시작")
    print(f"  Target Score : {TARGET_SCORE}점")
    print(f"  QA 문항 수   : {len(qa_sheet)}개")

    eval_history: Dict = {}

    # ── STEP 1: V1 Baseline ────────────────────────────
    _sep("STEP 1 : V1 Baseline")
    print("  방식: 표준 인덱스 + 표준 벡터 검색 (전략 미적용)")

    v1_index = "adp_v1"
    v1_docs  = preprocess_default(raw_docs, embedding_fn=bedrock_embed)
    _reset_index(client, v1_index)
    create_index_v1(client, v1_index)
    ingest_documents_v1(client, v1_index, v1_docs)
    client.indices.refresh(index=v1_index)

    v1_eval  = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_v1(client, v1_index, q))
    v1_score = _score(v1_eval)
    eval_history["v1"] = v1_eval
    _print_eval_row("V1", v1_score, v1_eval)

    # ── STEP 2: V2 (Keyword + Vector) ─────────────────
    _sep("STEP 2 : V2 (Keyword + Vector)")
    strategy_v2 = {**DEFAULT_STRATEGY, "index_version": "v2",
                   "use_keyword": True, "use_analyzer": False, "use_hyde": False, "use_dcr": False}
    print(f"  방식: 키워드 추출 + 벡터 검색")

    v2_docs  = preprocess_agentic(raw_docs, keyword_fn=extract_keywords, embedding_fn=bedrock_embed)
    v2_index = "adp_v2"
    _reset_index(client, v2_index)
    create_agent_index(client, v2_index, strategy_v2)
    ingest_documents_v2(client, v2_index, v2_docs)
    client.indices.refresh(index=v2_index)

    v2_eval  = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_agent(client, v2_index, q, strategy_v2))
    v2_score = _score(v2_eval)
    eval_history["v2"] = v2_eval
    _print_eval_row("V2", v2_score, v2_eval)

    # ── STEP 3: V3 (Keyword+Analyzer+DCR + HYDE) ──────
    _sep("STEP 3 : V3 (Analyzer+DCR + HYDE)")
    strategy_v3 = {**DEFAULT_STRATEGY, "index_version": "v3",
                   "use_keyword": True, "use_analyzer": True, "use_hyde": True, "use_dcr": True,
                   "analyzer_config": {"type": "nori", "decompound_mode": "mixed"}}
    print(f"  방식: Nori DCR 분석기 + HyDE 임베딩")

    all_keywords = extract_keywords(" ".join(d["text"] for d in raw_docs if d.get("text")))
    dcr_rules    = get_decompound_rules(all_keywords, client, v1_index)
    print(f"  DCR 규칙 수: {len(dcr_rules)}개")

    v3_docs  = preprocess_agentic(
        raw_docs,
        keyword_fn=extract_keywords,
        embedding_fn=bedrock_embed,
        use_analyzer_dcr=True,
        use_hyde=True,
        qa_sheet=qa_sheet,
        os_client=client,
        dcr_index=v1_index,
    )
    v3_index = "adp_v3"
    _reset_index(client, v3_index)
    create_agent_index(client, v3_index, strategy_v3, dcr_rules=dcr_rules)
    ingest_documents_v3(client, v3_index, v3_docs)
    client.indices.refresh(index=v3_index)

    v3_eval  = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_agent(client, v3_index, q, strategy_v3))
    v3_score = _score(v3_eval)
    eval_history["v3"] = v3_eval
    _print_eval_row("V3", v3_score, v3_eval)

    # ── 결과 요약 ──────────────────────────────────────
    _sep("결과 요약")
    for ver, ev in eval_history.items():
        _print_eval_row(ver.upper(), _score(ev), ev)

    best = max(eval_history, key=lambda v: eval_history[v].get("accuracy", 0))
    print(f"\n  🏆 최고 버전: {best.upper()} ({_score(eval_history[best])}점)")
    return _build_result(best, strategy_v3, eval_history)


def _build_result(
    best_version: str,
    strategy: Dict,
    eval_history: Dict,
    sanity_report: Optional[Dict] = None,
) -> Dict:
    return {
        "best_version":   best_version,
        "final_strategy": strategy,
        "eval_history":   {k: {"accuracy": v.get("accuracy"), "correct": v.get("correct"),
                               "total": v.get("total")} for k, v in eval_history.items()},
        "sanity_report":  sanity_report,
    }


# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        result = run_pipeline(
            user_prompt="기업 복지 규정 및 전결 권한 조회 시스템",
            raw_data_path="data/복지규정_demo.txt",
            qa_sheet_path="data/qa_sheet_demo.json",
        )
        print("\n" + "=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"\n❌ 파이프라인 실패: {exc}")
        print("  → OPENSEARCH_HOST/PORT, AWS 자격증명, .env 확인")
