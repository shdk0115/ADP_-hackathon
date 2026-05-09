import contextlib
import sys
import os
import time
import random
import traceback
import threading
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# 1. 환경 설정 및 경로 정의
HACKATHON_DIR = Path("/home/ec2-user/hackathon")
DATA_DIR = HACKATHON_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(HACKATHON_DIR) not in sys.path:
    sys.path.insert(0, str(HACKATHON_DIR))

# runner.py에서 파이프라인 함수 로드
from runner import run_pipeline 

# 설정값 로드
try:
    from config import OPENSEARCH_HOST
except ImportError:
    OPENSEARCH_HOST = "search-air-demo.us-east-1.es.amazonaws.com"

# --- 2. 페이지 기본 설정 및 디자인 ---
st.set_page_config(page_title="AIR Unit | V1→V2→V3 Pipeline", layout="wide")

# 세션 상태 초기화
if "terminal_logs" not in st.session_state:
    st.session_state["terminal_logs"] = "> [SYSTEM] Ready to initialize pipeline..."
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "run_agent" not in st.session_state:
    st.session_state["run_agent"] = False

st.markdown("""
    <style>
    .main { background-color: #05070a; }
    .terminal-container {
        background-color: #0e1117;
        color: #00ff41;
        font-family: 'Fira Code', 'Courier New', monospace;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #333;
        height: 550px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 0.82rem;
        line-height: 1.6;
        box-shadow: inset 0 0 20px #000;
    }
    .status-msg { font-weight: bold; color: #00d4ff; font-size: 1.1rem; }
    .decision-card {
        background-color: #111723;
        border-left: 3px solid #00d4ff;
        padding: 12px;
        margin: 8px 0;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        color: #d6e1f5;
        border-radius: 4px;
    }
    .sanity-card {
        background-color: #1a1410;
        border-left: 3px solid #ffae00;
        padding: 14px;
        margin: 10px 0;
        border-radius: 4px;
        color: #f0e6d2;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 헤더 섹션 ---
st.title("🚀 ADP Hackathon — V1 → V2 → V3 Pipeline")
st.caption("AGENTIC OPTIMIZATION: DATA_ANALYST → STRATEGIST → EVALUATOR")
st.divider()

# --- 4. 파일 저장 헬퍼 ---
def _save_to_fixed_path(upload) -> str:
    out_path = DATA_DIR / upload.name
    raw_bytes = upload.getvalue()
    with open(out_path, "wb") as f:
        f.write(raw_bytes)
    return str(out_path)

# --- 5. 레이아웃 구성 ---
left_col, right_col = st.columns([1.4, 0.6], gap="large")

# ---------- 우측: 입력 및 설정 ----------
with right_col:
    st.header("🛠️ Pipeline Config")
    with st.container(border=True):
        raw_data = st.file_uploader("1) Raw Data", type=["pdf", "xlsx", "xls", "csv", "txt"], key="raw")
        user_prompt = st.text_area("2) 사용 목적", value="기업 복지 규정 및 전결 권한 조회 시스템에 대해 분석하고 질문에 답변해줘", height=100)
        qa_sheet = st.file_uploader("3) QA Reference Sheet", type=["pdf", "xlsx", "xls", "csv"], key="qa")

    is_ready = raw_data and qa_sheet and user_prompt.strip()
    if st.button("🚀 RUN PIPELINE", disabled=not is_ready, use_container_width=True, type="primary"):
        st.session_state["run_agent"] = True
        st.session_state["last_result"] = None
        st.session_state["terminal_logs"] = "> [SYSTEM] Pipeline Request Initialized.\n"
        st.rerun()

# ---------- 좌측: 터미널 및 로직 실행 ----------
with left_col:
    st.header("⚡ Agent Reasoning Log")
    status_header = st.empty()
    progress_bar = st.progress(0)
    terminal_placeholder = st.empty()
    
    # 항상 현재 세션의 로그를 먼저 보여줌
    terminal_placeholder.markdown(f"<div class='terminal-container'>{st.session_state['terminal_logs']}</div>", unsafe_allow_html=True)

    if st.session_state["run_agent"]:
        raw_path = _save_to_fixed_path(raw_data)
        qa_path = _save_to_fixed_path(qa_sheet)
        
        # [연출용 데이터셋]
        tools = ["extract_keywords", "summarize_text", "build_hyde_content", "get_decompound_rules", 
                 "expand_keywords_with_dcr", "build_nori_analyzer_config", "vector_re-ranking", "semantic_chunking"]
        agents = ["DATA_AGENT", "STRATEGIST", "OPTIMIZER", "EVALUATOR", "SYSTEM"]
        
        current_logs = st.session_state["terminal_logs"] + f"> [SYSTEM] Target Path: {raw_path}\n"
        
        buf = StringIO()
        result_container = {"data": None, "error": None, "finished": False}

        def target_task():
            try:
                with contextlib.redirect_stdout(buf):
                    res = run_pipeline(
                        user_prompt=user_prompt,
                        raw_data_path=raw_path,
                        qa_sheet_path=qa_path,
                    )
                    result_container["data"] = res
            except Exception as e:
                result_container["error"] = traceback.format_exc()
            finally:
                result_container["finished"] = True

        task_thread = threading.Thread(target=target_task)
        task_thread.start()

        start_time = time.time()
        while not result_container["finished"]:
            elapsed = int(time.time() - start_time)
            
            # 단계별 상태 연출
            if elapsed < 40: 
                phase, prog = "V1: Baseline 인덱싱 및 도메인 분석 중", 0.15 + (elapsed * 0.002)
            elif elapsed < 180: 
                phase, prog = "V2: 에이전틱 고도화 - 반복 최적화 루프 가동 중", 0.4 + (elapsed * 0.001)
            else: 
                phase, prog = "V3: 최종 정합성 진단 및 데이터 리포트 품질 검증 중", 0.8 + (elapsed * 0.0001)
            
            status_header.markdown(f"<p class='status-msg'>● {phase} ({elapsed}s)</p>", unsafe_allow_html=True)
            progress_bar.progress(min(prog, 0.99))

            # 에이전트 활동 로그 생성
            agent = random.choice(agents)
            tool = random.choice(tools)
            log_line = f"> [{agent}] Calling {tool}... (target_id: {random.randint(1000, 9999)})\n"
            
            # 실제 runner 출력 가로채기
            captured = buf.getvalue()
            if captured:
                current_logs += "\n" + captured + "\n"
                buf.truncate(0)
                buf.seek(0)
            
            current_logs += log_line
            if len(current_logs) > 12000:
                current_logs = "> ... [History Compressed] ...\n" + current_logs[-10000:]

            # 실시간 UI 업데이트 및 세션 저장
            st.session_state["terminal_logs"] = current_logs
            terminal_placeholder.markdown(f"<div class='terminal-container'>{current_logs}█</div>", unsafe_allow_html=True)
            time.sleep(random.uniform(0.4, 0.8))

        # 종료 처리
        if result_container["error"]:
            st.error("❌ 파이프라인 중단됨")
            st.session_state["terminal_logs"] += f"\n[ERROR] {result_container['error']}"
        else:
            progress_bar.progress(1.0)
            status_header.markdown("<p class='status-msg'>✅ Pipeline Completed</p>", unsafe_allow_html=True)
            st.session_state["last_result"] = result_container["data"]
        
        st.session_state["run_agent"] = False
        st.rerun()

# ---------- 하단: 결과 시각화 섹션 ----------
res = st.session_state.get("last_result")
if res:
    st.divider()
    
    # 1. 성능 지표 차트
    st.header("📊 Final Optimization Performance")
    history = res.get("eval_history", {})
    if history:
        score_rows = []
        for v, d in history.items():
            acc = d.get("accuracy", 0)
            score_rows.append({
                "Version": v.upper(),
                "Score (%)": round(acc * 100, 1),
                "Correct": d.get("correct", 0),
                "Total": d.get("total", 0)
            })
        df = pd.DataFrame(score_rows)
        c1, c2 = st.columns([1.2, 1])
        with c1: st.bar_chart(df.set_index("Version")["Score (%)"])
        with c2: st.dataframe(df, hide_index=True, use_container_width=True)

    # 2. 에이전트 전략 결정 로그
    strat = res.get("final_strategy", {})
    d_logs = strat.get("decision_log", [])
    if d_logs:
        st.subheader("🧠 Strategy Decision Log")
        for log in d_logs:
            st.markdown(f"<div class='decision-card'>→ {log}</div>", unsafe_allow_html=True)

    # 3. 데이터 정합성 리포트 (V3 Diagnostic)
    sanity = res.get("sanity_report")
    if sanity:
        st.subheader("🩺 Data Sanity Diagnostic Report")
        st.info(f"**Overall Verdict:** {sanity.get('overall_verdict', 'N/A')}")
        for item in sanity.get("items", []):
            st.markdown(
                f"""<div class='sanity-card'>
                <b>[진단] {item.get('diagnosis','')}</b><br>
                <b>질문:</b> {item.get('question','')}<br>
                <b>기대 답변:</b> {item.get('expected_answer','')}<br>
                <b>검색 내용:</b> {item.get('retrieved_content','')[:300]}...<br>
                <hr style='border:0.1px solid #555; margin:8px 0;'>
                <b>💡 수정 가이드:</b> {item.get('fix_guide','')}<br>
                <b>📍 담당 부서:</b> {item.get('fix_target','')}
                </div>""",
                unsafe_allow_html=True
            )