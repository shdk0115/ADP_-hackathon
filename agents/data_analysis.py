import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import boto3

from config import AWS_REGION, BEDROCK_TEXT_MODEL
from prompts.data_profile import DATA_PROFILE_PROMPT
from prompts.data_sanity import DATA_SANITY_PROMPT
from prompts.strategy_decision import RETRY_PROMPT, STRATEGY_DECISION_PROMPT
from prompts.strategy_refinement import STRATEGY_REFINEMENT_PROMPT

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Fallback: LLM 3회 실패 시 사용되는 기본값
# ──────────────────────────────────────────────
DEFAULT_ANALYSIS_RESULT: Dict = {
    "index_version": "v2",
    "use_keyword": True,
    "use_vector": True,
    "use_hybrid": True,
    "use_analyzer": True,
    "use_hyde": False,
    "use_dcr": False,
    "fields": {
        "meta_data": "object",
        "contents": "text",
        "keywords": "text",
        "embedding": "knn_vector",
    },
    "analyzer_config": {"type": "nori", "decompound_mode": "mixed"},
    "chunking_strategy": "fixed",
    "decision_log": ["Fallback default applied — LLM analysis failed after max retries."],
}

# ──────────────────────────────────────────────
# Validation Rules
# ──────────────────────────────────────────────
VALIDATION_RULES: List[Tuple] = [
    (
        lambda s: s.get("use_vector") is True,
        "use_vector must always be true",
    ),
    (
        lambda s: s.get("index_version") in {"v1", "v2", "v3"},
        "index_version must be 'v1', 'v2', or 'v3'",
    ),
    (
        lambda s: s.get("index_version") != "v3"
        or (s.get("use_analyzer") is True and s.get("use_hyde") is True),
        "v3 requires both use_analyzer and use_hyde to be true",
    ),
    (
        lambda s: not s.get("use_hybrid") or s.get("use_keyword") is True,
        "use_hybrid requires use_keyword to be true",
    ),
    (
        lambda s: isinstance(s.get("decision_log"), list)
        and len(s.get("decision_log", [])) > 0,
        "decision_log must be a non-empty list of strings",
    ),
    (
        lambda s: not s.get("use_keyword")
        or "keywords" in s.get("fields", {}),
        "fields must include 'keywords' when use_keyword is true",
    ),
]


def _validate_strategy(strategy: Dict) -> Optional[str]:
    for rule_fn, error_msg in VALIDATION_RULES:
        try:
            if not rule_fn(strategy):
                return error_msg
        except Exception:
            return f"Rule evaluation error: {error_msg}"
    return None


# ──────────────────────────────────────────────
# LLM 클라이언트
# ──────────────────────────────────────────────
def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _call_llm(prompt: str) -> str:
    client = _get_bedrock_client()
    model_id = BEDROCK_TEXT_MODEL

    # Amazon Titan Text
    if "titan" in model_id.lower():
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "temperature": 0.0,
                "topP": 1.0,
            },
        })
        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"].strip()

    # Anthropic Claude on Bedrock
    if "claude" in model_id.lower() or "anthropic" in model_id.lower():
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()

    raise ValueError(f"Unsupported model: {model_id}")


def _parse_json_response(raw: str) -> Dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    return json.loads(cleaned)


# ──────────────────────────────────────────────
# Step 1: Data Profiling
# ──────────────────────────────────────────────
def _run_data_profiling(
    user_prompt: str,
    data_sample: str,
    max_retry: int,
    domain_hint: str = "",
    few_shot: str = "",
) -> Dict:
    full_prompt = DATA_PROFILE_PROMPT.format(
        user_prompt=user_prompt + (f"\n\n{domain_hint}" if domain_hint else ""),
        data_sample=data_sample[:3000],
    )
    if few_shot:
        full_prompt += f"\n\n## Reference\n{few_shot}"

    last_error = ""
    raw = ""
    for attempt in range(1, max_retry + 1):
        try:
            if attempt > 1:
                full_prompt = RETRY_PROMPT.format(
                    error_message=last_error,
                    previous_response=raw,
                ) + full_prompt
            raw = _call_llm(full_prompt)
            profile = _parse_json_response(raw)
            logger.info("[Profiling] Success on attempt %d", attempt)
            return profile
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = str(e)
            logger.warning("[Profiling] Attempt %d failed: %s", attempt, last_error)

    raise RuntimeError(f"Data profiling failed after {max_retry} retries. Last error: {last_error}")


# ──────────────────────────────────────────────
# Step 2: Strategy Decision
# ──────────────────────────────────────────────
def _run_strategy_decision(profile: Dict, max_retry: int) -> Dict:
    base_prompt = STRATEGY_DECISION_PROMPT.format(
        data_profile=json.dumps(profile, ensure_ascii=False, indent=2)
    )

    prompt = base_prompt
    raw = ""
    last_error = ""

    for attempt in range(1, max_retry + 1):
        try:
            if attempt > 1:
                prompt = RETRY_PROMPT.format(
                    error_message=last_error,
                    previous_response=raw,
                ) + base_prompt

            raw = _call_llm(prompt)
            strategy = _parse_json_response(raw)

            error_msg = _validate_strategy(strategy)
            if error_msg:
                last_error = error_msg
                logger.warning("[Strategy] Attempt %d validation failed: %s", attempt, error_msg)
                continue

            logger.info("[Strategy] Success on attempt %d", attempt)
            return strategy

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = str(e)
            logger.warning("[Strategy] Attempt %d parse failed: %s", attempt, last_error)

    raise RuntimeError(
        f"Strategy decision failed after {max_retry} retries. Last error: {last_error}"
    )


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def analyze_data_with_retry(
    user_prompt: str,
    data_sample: str = "",
    max_retry: int = 3,
    domain_hint: str = "",
    few_shot: str = "",
) -> Dict:
    """
    데이터 샘플 + 사용자 프롬프트 → Strategy JSON

    Flow:
      1. Data Profiling  (LLM, max_retry)
      2. Strategy Decision (LLM, max_retry + validation)
      3. Fallback → DEFAULT_ANALYSIS_RESULT (모두 실패 시)

    Returns:
      {
        "profile":  { ... },   # Step 1 결과 (데모 화면 표시용)
        "strategy": { ... },   # Step 2 결과 (Paul에게 전달)
      }
    """
    sample = data_sample or user_prompt

    # Step 1: Profiling
    try:
        profile = _run_data_profiling(
            user_prompt=user_prompt,
            data_sample=sample,
            max_retry=max_retry,
            domain_hint=domain_hint,
            few_shot=few_shot,
        )
    except RuntimeError as e:
        logger.error("[analyze] Profiling failed, using fallback. Reason: %s", e)
        return {"profile": {}, "strategy": DEFAULT_ANALYSIS_RESULT.copy()}

    # Step 2: Strategy Decision
    try:
        strategy = _run_strategy_decision(profile=profile, max_retry=max_retry)
    except RuntimeError as e:
        logger.error("[analyze] Strategy decision failed, using fallback. Reason: %s", e)
        return {"profile": profile, "strategy": DEFAULT_ANALYSIS_RESULT.copy()}

    return {"profile": profile, "strategy": strategy}


# ──────────────────────────────────────────────
# Strategy Refinement (V2 실패 후 전략 수정)
# ──────────────────────────────────────────────
def refine_strategy_from_errors(
    current_strategy: Dict,
    error_report: Dict,
    max_retry: int = 3,
) -> Dict:
    """
    V2 이후 평가 실패 시 에러 분석 결과로 전략을 수정.

    Args:
        current_strategy : 현재까지 사용한 Strategy JSON
        error_report     : 평가 실패 분석 결과
          {
            "score": float,
            "failed_questions": [...],
            "error_patterns": [
              "CONDITIONAL_REASONING_FAIL",
              "ENTITY_MAPPING_FAIL",
              ...
            ]
          }

    Returns:
        수정된 Strategy JSON (decision_log에 변경 근거 포함)
    """
    base_prompt = STRATEGY_REFINEMENT_PROMPT.format(
        current_strategy=json.dumps(current_strategy, ensure_ascii=False, indent=2),
        error_report=json.dumps(error_report, ensure_ascii=False, indent=2),
    )

    prompt = base_prompt
    raw = ""
    last_error = ""

    for attempt in range(1, max_retry + 1):
        try:
            if attempt > 1:
                prompt = RETRY_PROMPT.format(
                    error_message=last_error,
                    previous_response=raw,
                ) + base_prompt

            raw = _call_llm(prompt)
            refined = _parse_json_response(raw)

            # Validation 검사
            error_msg = _validate_strategy(refined)
            if error_msg:
                last_error = error_msg
                logger.warning(
                    "[Refinement] Attempt %d validation failed: %s", attempt, error_msg
                )
                continue

            # index_version 다운그레이드 방지
            version_order = {"v1": 1, "v2": 2, "v3": 3}
            curr_v = version_order.get(current_strategy.get("index_version", "v1"), 1)
            new_v  = version_order.get(refined.get("index_version", "v1"), 1)
            if new_v < curr_v:
                last_error = (
                    f"index_version downgrade not allowed: "
                    f"{current_strategy['index_version']} → {refined['index_version']}"
                )
                logger.warning("[Refinement] Attempt %d: %s", attempt, last_error)
                continue

            logger.info("[Refinement] Success on attempt %d", attempt)
            return refined

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = str(e)
            logger.warning("[Refinement] Attempt %d parse failed: %s", attempt, last_error)

    # Fallback: 현재 전략 유지 + log 추가
    logger.error("[Refinement] Failed after %d retries, keeping current strategy", max_retry)
    fallback = current_strategy.copy()
    fallback["decision_log"] = current_strategy.get("decision_log", []) + [
        f"Refinement failed after {max_retry} retries — current strategy retained."
    ]
    return fallback


# ──────────────────────────────────────────────
# Data Sanity Diagnosis (Max Retry 소진 후 호출)
# ──────────────────────────────────────────────
def diagnose_data_sanity(
    failed_items: List[Dict],
    retrieved_excerpts: List[Dict],
    max_retry: int = 3,
) -> Dict:
    """
    Max Retry 소진 후에도 목표 점수 미달 시 호출.
    QA 정답 vs 검색 결과 불일치 원인을 진단.

    Args:
        failed_items: 실패한 QA 항목 리스트
          [{ "question": str, "expected_answer": str }, ...]
        retrieved_excerpts: 각 질문별 검색된 내용
          [{ "question": str, "retrieved_content": str }, ...]

    Returns:
        {
          "overall_verdict": "PIPELINE_ISSUE" | "DATA_ISSUE" | "MIXED",
          "items": [...],
          "summary": str
        }
    """
    prompt = DATA_SANITY_PROMPT.format(
        failed_items=json.dumps(failed_items, ensure_ascii=False, indent=2),
        retrieved_excerpts=json.dumps(retrieved_excerpts, ensure_ascii=False, indent=2),
    )

    last_error = ""
    raw = ""
    for attempt in range(1, max_retry + 1):
        try:
            if attempt > 1:
                prompt = RETRY_PROMPT.format(
                    error_message=last_error,
                    previous_response=raw,
                ) + prompt
            raw = _call_llm(prompt)
            result = _parse_json_response(raw)

            # 최소 검증
            if "overall_verdict" not in result or "items" not in result:
                raise ValueError("Missing required fields: overall_verdict or items")

            logger.info("[DataSanity] Success on attempt %d", attempt)
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = str(e)
            logger.warning("[DataSanity] Attempt %d failed: %s", attempt, last_error)

    # Fallback
    logger.error("[DataSanity] Failed after %d retries", max_retry)
    return {
        "overall_verdict": "MIXED",
        "items": [],
        "summary": "데이터 정합성 진단 실패 — 수동 검토 필요",
    }
