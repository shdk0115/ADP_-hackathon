import json
from pathlib import Path
from typing import Dict, List, Optional

from agents.data_analysis import (
    analyze_data_with_retry,
    diagnose_data_sanity,
    refine_strategy_from_errors,
)
from config import (
    MAX_ANALYSIS_RETRY,
    MAX_PIPELINE_RETRY,
    OPENSEARCH_HOST,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_USE_SSL,
    TARGET_SCORE,
)
from evaluation.error_analysis import analyze_errors
from evaluation.evaluator import evaluate_qa_sheet
from opensearch.agentic_index import create_agent_index
from opensearch.agentic_search import agent_hybrid_search
from opensearch.default_index import create_index_v1
from opensearch.default_search import default_hybrid_search
from opensearch.ingest import ingest_documents_default
from prompts.domain.corporate import CORPORATE_DOMAIN_HINT, CORPORATE_FEW_SHOT
from tools.add_dcr import get_decompound_rules
from tools.keyword_extraction import extract_keywords
from tools.preprocessing import hash_embed, load_raw_documents, preprocess_agentic, preprocess_default

try:
    from opensearchpy import OpenSearch
except ImportError:
    OpenSearch = None

BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    return str(p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve())


# ──────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────
def _load_qa_sheet(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists() or not p.read_text(encoding="utf-8").strip():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


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
def _keyword_exposure_count(client, index_name, doc_id, query_tokens):
    tv = client.termvectors(index=index_name, id=doc_id, fields=["keywords"], term_statistics=True)
    terms = tv.get("term_vectors", {}).get("keywords", {}).get("terms", {})
    return sum(terms.get(tok, {}).get("term_freq", 0) for tok in query_tokens)


def _search_v1(client, index_name, query, size=5):
    qv = hash_embed(query)
    return default_hybrid_search(client, index_name, query, qv, size=size)


def _search_agent(client, index_name, query, strategy, size=5):
    qv = hash_embed(query)
    use_keyword = strategy.get("use_keyword", False)
    hits = agent_hybrid_search(client, index_name, query, qv, use_keyword=use_keyword, size=size * 2)
    if use_keyword:
        query_tokens = extract_keywords(query, top_k=10)
        for hit in hits:
            freq = _keyword_exposure_count(client, index_name, hit["_id"], query_tokens)
            hit["_score"] = float(hit.get("_score", 0.0)) + (1.5 * freq)
        hits.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return hits[:size]


# ──────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────
def run_pipeline(
    user_prompt: str,
    raw_data_path: str = "data/복지규정_demo.txt",
    qa_sheet_path: str = "data/qa_sheet_demo.json",
) -> Dict:

    raw_path = _resolve_data_path(raw_data_path)
    qa_path = _resolve_data_path(qa_sheet_path)

    raw_docs = load_raw_documents(raw_path)
    qa_sheet = _load_qa_sheet(qa_path)
    client   = _get_client()

    _sep("ADP 해커톤 파이프라인 시작")
    print(f"  Target Score     : {TARGET_SCORE}점")
    print(f"  Max Pipeline Retry: {MAX_PIPELINE_RETRY}회")
    print(f"  QA 문항 수       : {len(qa_sheet)}개")

    # ── STEP 1: 데이터 분석 ────────────────────────────
    _sep("STEP 1 : 데이터 분석")

    with open(raw_path, encoding="utf-8") as f:
        data_sample = f.read()

    analysis_result = analyze_data_with_retry(
        user_prompt=user_prompt,
        data_sample=data_sample,
        max_retry=MAX_ANALYSIS_RETRY,
        domain_hint=CORPORATE_DOMAIN_HINT,
        few_shot=CORPORATE_FEW_SHOT,
    )
    profile  = analysis_result["profile"]
    strategy = analysis_result["strategy"]

    print(f"  도메인    : {profile.get('domain', {}).get('type', 'unknown')}")
    print(f"  언어      : {profile.get('language', {}).get('primary', 'unknown')}")
    print(f"  복잡도    : {profile.get('content', {}).get('semantic_complexity', 'unknown')}")
    print()
    _print_decision_log(strategy.get("decision_log", []))
    print(f"\n  확정 전략 : {strategy['index_version']} | "
          f"keyword:{strategy['use_keyword']} | "
          f"analyzer:{strategy['use_analyzer']} | "
          f"hyde:{strategy['use_hyde']}")

    # ── STEP 2: V1 Baseline ────────────────────────────
    _sep("STEP 2 : V1 Baseline")
    print("  방식: 표준 인덱스 + 표준 벡터 검색 (전략 미적용)")

    v1_index = "adp_v1"
    default_docs = preprocess_default(raw_docs, embedding_fn=hash_embed)
    _reset_index(client, v1_index)
    create_index_v1(client, v1_index)
    ingest_documents_default(client, v1_index, default_docs)
    client.indices.refresh(index=v1_index)

    v1_eval  = evaluate_qa_sheet(qa_sheet, search_fn=lambda q: _search_v1(client, v1_index, q))
    v1_score = _score(v1_eval)
    _print_eval_row("V1", v1_score, v1_eval)

    if v1_score >= TARGET_SCORE:
        _sep("완료")
        print(f"  ✅ V1에서 목표 달성! ({v1_score}점)")
        return _build_result("v1", strategy, {"v1": v1_eval})

    # ── STEP 3: 피드백 기반 Agentic 루프 ───────────────
    eval_history = {"v1": v1_eval}
    current_strategy = strategy
    best_score = v1_score
    best_version = "v1"

    for retry in range(1, MAX_PIPELINE_RETRY + 1):
        version = f"v{retry + 1}"
        _sep(f"STEP {retry + 2} : {version.upper()} Agentic (retry {retry}/{MAX_PIPELINE_RETRY})")

        # 에러 분석
        error_analysis = analyze_errors(eval_history[f"v{retry}"]["errors"])
        failed_qs = [e for e in eval_history[f"v{retry}"]["errors"]]
        error_patterns = error_analysis.get("error_patterns", [])
        print(f"  실패 문항  : {[e.get('question','')[:20] for e in failed_qs]}")
        print(f"  에러 패턴  : {error_patterns}")

        # 전략 수정 (retry 2회차부터 refine 적용)
        if retry > 1:
            print()
            current_strategy = refine_strategy_from_errors(
                current_strategy=current_strategy,
                error_report={
                    "score": best_score,
                    "failed_questions": failed_qs,
                    "error_patterns": error_patterns,
                },
                max_retry=MAX_ANALYSIS_RETRY,
            )
            print("  [AI 전략 수정 근거]")
            for log in current_strategy.get("decision_log", []):
                print(f"  → {log}")

        # 전처리 + 인덱스 + 적재
        use_dcr = current_strategy.get("use_dcr", False)
        use_hyde = current_strategy.get("use_hyde", False)

        # Build DCR rules: use default index as nori analyzer source
        dcr_rules = []
        if use_dcr:
            all_keywords = extract_keywords(" ".join(
                d["text"] for d in raw_docs if d.get("text")
            ))
            dcr_rules = get_decompound_rules(all_keywords, client, v1_index)

        agent_docs = preprocess_agentic(
            raw_docs,
            keyword_fn=lambda x: extract_keywords(x),
            embedding_fn=hash_embed,
            use_analyzer_dcr=use_dcr,
            use_hyde=use_hyde,
            qa_sheet=qa_sheet,
            os_client=client,
            dcr_index=v1_index,
        )
        agent_index = f"adp_{version}"
        _reset_index(client, agent_index)
        create_agent_index(client, agent_index, current_strategy, dcr_rules=dcr_rules)
        ingest_documents_default(client, agent_index, agent_docs)
        client.indices.refresh(index=agent_index)

        # 평가
        agent_eval  = evaluate_qa_sheet(
            qa_sheet,
            search_fn=lambda q: _search_agent(client, agent_index, q, current_strategy),
        )
        agent_score = _score(agent_eval)
        eval_history[version] = agent_eval
        _print_eval_row(version.upper(), agent_score, agent_eval)

        if agent_score > best_score:
            best_score   = agent_score
            best_version = version

        if agent_score >= TARGET_SCORE:
            _sep("완료")
            print(f"  ✅ {version.upper()}에서 목표 달성! ({agent_score}점)")
            return _build_result(version, current_strategy, eval_history)

    # ── STEP 마지막: Data Sanity ────────────────────────
    _sep("⚠️  Data Sanity Diagnosis")
    print(f"  Max Retry({MAX_PIPELINE_RETRY}회) 소진. 최고 점수: {best_score}점 ({best_version})")
    print("  원천 데이터 오류 여부 진단 중...")

    last_eval   = eval_history[best_version]
    failed_items = [
        {"question": e.get("question", ""), "expected_answer": e.get("expected", "")}
        for e in last_eval.get("errors", [])
    ]
    retrieved = [
        {"question": e.get("question", ""), "retrieved_content": e.get("got", "")}
        for e in last_eval.get("errors", [])
    ]

    sanity_report = diagnose_data_sanity(
        failed_items=failed_items,
        retrieved_excerpts=retrieved,
        max_retry=MAX_ANALYSIS_RETRY,
    )

    print(f"\n  overall_verdict: {sanity_report['overall_verdict']}")
    for item in sanity_report.get("items", []):
        _sep()
        print(f"  진단      : [{item['diagnosis']}]")
        print(f"  질문      : {item['question']}")
        print(f"  기대값    : {item['expected_answer']}")
        print(f"  검색값    : {item['retrieved_content']}")
        print(f"  ⚠️  수정가이드: {item['fix_guide']}")
        print(f"  담당자    : {item['fix_target']}")
    _sep()
    print(f"  📋 {sanity_report['summary']}")

    return _build_result(best_version, current_strategy, eval_history, sanity_report)


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
