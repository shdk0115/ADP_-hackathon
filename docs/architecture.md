# 아키텍처 설계서

**프로젝트명:** ADP — Agentic OpenSearch Indexing Pipeline (V1 → V2 → V3)
**팀:** AIR Unit
**작성일:** 2026-05-09
**버전:** 1.0

---

## 1. 시스템 개요

### 1.1 목적
사용자가 업로드한 도메인 문서(PDF / Excel / CSV / TXT)와 평가용 QA 시트(Excel / PDF)를 입력으로 받아,
**LLM 기반 데이터 분석 에이전트가 OpenSearch 인덱싱 전략을 자동으로 결정**하고,
세 단계(V1 → V2 → V3)에 걸쳐 점진적으로 검색 정확도를 향상시키는 RAG 파이프라인.

### 1.2 핵심 가치 제안
- **자동 전략 결정**: 데이터를 보고 LLM이 인덱스 버전, 분석기, 키워드/HyDE 적용 여부를 결정
- **점진적 개선의 가시화**: V1(베이스라인) → V2(키워드 보강) → V3(형태소 + DCR + HyDE) 정확도 향상이 시연으로 증명됨
- **데이터 정합성 진단**: 목표 점수 미달 시 LLM이 데이터 자체의 결함을 자동 진단

### 1.3 기술 스택
| 계층 | 기술 |
|---|---|
| UI | Streamlit |
| LLM | Amazon Bedrock — Claude Sonnet 4.6 (`us.anthropic.claude-sonnet-4-6`) |
| Embedding | Amazon Bedrock — Titan Embed Text v2 (`amazon.titan-embed-text-v2:0`, 1024 dim) |
| Vector / Text Search | OpenSearch (k-NN + BM25 + nori 형태소 분석기) |
| Document Loader | pypdf, openpyxl, pandas |
| Runtime | Python 3.9 |

---

## 2. 전체 아키텍처

### 2.1 시스템 컨텍스트 다이어그램

```
                ┌─────────────────────────────────────────────┐
                │                  USER                        │
                │   (업무 담당자 / RAG 엔지니어 / 데이터 오너)        │
                └──────────────────────┬──────────────────────┘
                                       │ (1) 파일 업로드
                                       ▼
                ┌─────────────────────────────────────────────┐
                │           Streamlit UI                       │
                │  ┌──────────┐  ┌────────────┐  ┌─────────┐  │
                │  │ 입력 폼   │  │ 실시간 로그 │  │ 결과 시각화 │  │
                │  └──────────┘  └────────────┘  └─────────┘  │
                └──────────────────────┬──────────────────────┘
                                       │ (2) run_pipeline()
                                       ▼
        ┌──────────────────────────────────────────────────────────┐
        │              Hackathon Pipeline (Python)                  │
        │                                                           │
        │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
        │  │  Loader      │──▶│ Data Analyst │──▶│  Strategy    │   │
        │  │ (PDF/Excel)  │   │  Agent (LLM) │   │   JSON       │   │
        │  └──────────────┘   └──────────────┘   └──────┬───────┘   │
        │                                                │           │
        │           ┌────────────────────────────────────┴────────┐  │
        │           ▼                ▼                ▼           │  │
        │     ┌──────────┐    ┌──────────┐    ┌──────────┐        │  │
        │     │ V1 BUILD │    │ V2 BUILD │    │ V3 BUILD │        │  │
        │     │ (default)│    │ (kw+vec) │    │(nori+DCR │        │  │
        │     │          │    │          │    │  +HyDE)  │        │  │
        │     └────┬─────┘    └────┬─────┘    └────┬─────┘        │  │
        │          │               │                │              │  │
        │          ▼               ▼                ▼              │  │
        │     ┌─────────────────────────────────────────────┐      │  │
        │     │           Evaluator (QA Sheet)              │      │  │
        │     │     accuracy + error patterns + sanity      │      │  │
        │     └─────────────────────────────────────────────┘      │  │
        └────────┬───────────────────────────────────┬─────────────┘
                 │                                   │
                 ▼ (Bedrock invoke)                  ▼ (REST API)
        ┌─────────────────────┐         ┌─────────────────────────┐
        │   Amazon Bedrock    │         │      OpenSearch          │
        │  • Claude Sonnet 4.6│         │  • k-NN vector index     │
        │  • Titan Embed v2   │         │  • nori_custom analyzer  │
        └─────────────────────┘         └─────────────────────────┘
```

### 2.2 V1 → V2 → V3 단계별 비교

```
┌────────────┬───────────────────┬───────────────────┬───────────────────────┐
│            │       V1          │       V2          │         V3            │
├────────────┼───────────────────┼───────────────────┼───────────────────────┤
│ 전처리      │ 단순 chunking      │ + LLM 키워드 추출  │ + Nori DCR + HyDE     │
│            │                   │                   │   (요약 + Q 매칭)       │
├────────────┼───────────────────┼───────────────────┼───────────────────────┤
│ 인덱스      │ TEXT(text)        │ TEXT + keyword    │ TEXT + keyword        │
│            │ embedding(knn)    │ embedding(knn)    │ + nori_custom          │
│            │                   │                   │ + DCR user_dict        │
│            │                   │                   │ embedding(knn)         │
├────────────┼───────────────────┼───────────────────┼───────────────────────┤
│ 검색        │ match(TEXT)       │ match(keyword)    │ knn(embedding) 1차    │
│            │ + knn(embedding)  │ + knn(embedding)  │ + termvectors(keyword)│
│            │                   │                   │   re-rank             │
├────────────┼───────────────────┼───────────────────┼───────────────────────┤
│ 강점        │ 본문 직접 매칭      │ 약어/도메인        │ 형태소 변형 +          │
│            │                   │ 용어 보강          │ 의미 확장              │
├────────────┼───────────────────┼───────────────────┼───────────────────────┤
│ 가설 점수    │  ~40%             │  ~70%             │  ~95%                 │
└────────────┴───────────────────┴───────────────────┴───────────────────────┘
```

---

## 3. 컴포넌트별 설계

### 3.1 입력 로더 (`tools/file_reader.py`)

| 함수 | 입력 | 처리 |
|---|---|---|
| `read_raw_data` | `.pdf / .xlsx / .csv / .txt` | 확장자 분기 → 평문 텍스트 반환 |
| `read_qa_sheet_excel` | `.xlsx / .csv` | pandas로 읽고 컬럼명 정규화 (`질문 (User Question)` → `question`) |
| `read_qa_sheet_pdf` | `.pdf` | pypdf로 텍스트 추출 → **LLM에 QA 쌍 자동 추출 위임** |

**한국어 인코딩 폴백**: utf-8 / utf-8-sig / cp949 / euc-kr 순회

### 3.2 Data Analyst Agent (`agents/data_analysis.py`)

```
┌──────────────────────────────────────────────────────────────┐
│                   Data Analyst Agent                          │
│                                                              │
│   사용자 프롬프트 + 데이터 샘플                                  │
│         │                                                    │
│         ▼                                                    │
│   ┌──────────────────┐                                       │
│   │ Step 1:           │   ko_ratio, has_tables,              │
│   │ Data Profiling    │   keyword_density, semantic_         │
│   │ (Claude)          │   complexity, domain_type 등         │
│   └────────┬─────────┘                                       │
│            │ profile JSON                                    │
│            ▼                                                 │
│   ┌──────────────────┐                                       │
│   │ Step 2:           │   결정 규칙(use_analyzer/use_keyword/  │
│   │ Strategy          │   use_hyde/use_dcr/index_version)     │
│   │ Decision (Claude) │                                       │
│   └────────┬─────────┘                                       │
│            │ strategy JSON + decision_log                    │
│            ▼                                                 │
│   ┌──────────────────┐                                       │
│   │ Validator         │   v3 → use_analyzer & use_hyde 강제   │
│   │ + Retry (max 3회) │   use_hybrid → use_keyword 강제       │
│   └────────┬─────────┘                                       │
│            │ Pass                                            │
│            ▼                                                 │
│       Strategy → 파이프라인                                    │
│                                                              │
│   (Pipeline 점수 미달 시) ───▶ Strategy Refinement (LLM)        │
│   (Max retry 소진 시)    ───▶ Data Sanity Diagnosis (LLM)     │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 전처리 (`tools/preprocessing.py`)

| 함수 | 출력 키 | 비고 |
|---|---|---|
| `preprocess_default` (V1) | `meta_info`, `content`, `embedding` | 단순 sentence chunking + Titan v2 임베딩 |
| `preprocess_agentic` (V2/V3) | `metainfo`, `TEXT`, `keyword`, `embedding` | LLM 키워드 + (V3: HyDE 본문 합성, DCR 토큰 확장) |

### 3.4 인덱스 (`opensearch/`)

| 인덱스 | 매핑 |
|---|---|
| **V1** (`adp_v1`) | `metafield`(object) · `TEXT`(text) · `embedding`(knn_vector, 1024) |
| **V2** (`adp_v2`) | + `keyword`(text) |
| **V3** (`adp_v3`) | + `nori_custom` analyzer + DCR `user_dictionary_rules` |

**V3 analyzer 설정 예시**
```json
{
  "tokenizer": {
    "nori_tokenizer": {
      "type": "nori_tokenizer",
      "decompound_mode": "mixed",
      "user_dictionary_rules": ["AI 인프라 부서", "긴급 통제 센터", ...]
    }
  }
}
```

### 3.5 검색 (`opensearch/agentic_search.py`)

```
┌─ V1 ─────────────────────────────┐
│  bool.should:                    │
│   ├─ match(TEXT, boost 0.5)      │
│   └─ knn(embedding, boost 0.5)   │
│  → top-1 hit                     │
└──────────────────────────────────┘

┌─ V2 ─────────────────────────────┐
│  bool.should:                    │
│   ├─ match(keyword, boost 0.5)   │
│   └─ knn(embedding, boost 0.5)   │
│  → top-1 hit                     │
└──────────────────────────────────┘

┌─ V3 ─────────────────────────────┐
│  1. Query → nori_custom 토큰화    │
│  2. knn(embedding, k=size×2)     │
│  3. for each hit:                │
│       termvectors(keyword)       │
│       → tf 합산                   │
│  4. score = vec_score + tf_count │
│  5. re-rank → top-size           │
└──────────────────────────────────┘
```

### 3.6 평가 (`evaluation/`)

| 모듈 | 역할 |
|---|---|
| `evaluator.py` | 각 QA 질문 → search → top-1의 TEXT 안에 expected_answer가 substring으로 있는지 확인 |
| `error_analysis.py` | 실패 질문 패턴 분류 (DATE_BOUNDARY / ENTITY_MAPPING / CONDITIONAL_REASONING / TABLE_RETRIEVAL / VECTOR_ONLY_INSUFFICIENT) |

---

## 4. 데이터 흐름 (End-to-End)

```
[사용자 업로드]
   raw_data: 복지규정.pdf
   qa_sheet: qa_시트.xlsx
        │
        ▼
[Streamlit] tempfile에 원본 파일명 보존하여 저장
        │
        ▼
[run_pipeline]
   ├─ load_raw_documents → raw_docs[{source, text, meta_data}]
   ├─ _load_qa_sheet → qa_sheet[{id, question, expected_answer}]
   │
   ├─ STEP 1: 데이터 분석
   │     Profile(JSON) → Strategy(JSON, decision_log)
   │
   ├─ STEP 2: V1 빌드
   │     preprocess_default → V1 ingest → V1 search → V1 score
   │
   ├─ STEP 3: V2 빌드
   │     preprocess_agentic(use_keyword=True)
   │       ↳ LLM 키워드 추출 → keyword 필드에 본문+키워드 통합
   │     V2 ingest → V2 search → V2 score
   │
   ├─ STEP 4: V3 빌드
   │     preprocess_agentic(use_hyde=True, use_dcr=True)
   │       ↳ LLM HyDE: 본문 + 요약 + 매칭 질문 합성
   │       ↳ LLM 키워드 + 본문 → DCR 사전 자동 생성
   │     V3 ingest (nori_custom + DCR) → V3 search (knn + re-rank) → V3 score
   │
   ├─ (선택) Strategy Refinement: 점수 부진 시 전략 재결정
   ├─ (선택) Data Sanity Diagnosis: 데이터 결함 자동 진단
   │
   └─ Result: { eval_history, final_strategy, sanity_report }
        │
        ▼
[Streamlit 시각화]
   ├─ V1/V2/V3 점수 막대 그래프
   ├─ Strategy decision_log 카드
   └─ Data Sanity 진단 카드 (DATA_MISMATCH / RETRIEVAL_FAILURE 등)
```

---

## 5. 외부 의존성

| 서비스 | 용도 | 호출 빈도 |
|---|---|---|
| **Bedrock Claude Sonnet 4.6** | Profiling / Strategy / 키워드 추출 / HyDE 요약 / Q 매칭 / Sanity 진단 | 파이프라인당 ~30회 (캐시 포함 시 감소) |
| **Bedrock Titan Embed v2** | 본문 chunk + query 임베딩 (1024 dim) | 파이프라인당 ~50회 (LRU 캐시 적용) |
| **OpenSearch** | k-NN + BM25 + nori 인덱싱/검색 | V1/V2/V3 각각 1 인덱스 |

**고가용성 / 비용 고려**
- Titan v2 임베딩: `functools.lru_cache(maxsize=4096)`로 동일 텍스트 재호출 차단 (~2400× 속도 향상 측정)
- LLM 호출 실패 시 `DEFAULT_ANALYSIS_RESULT` fallback 적용
- OpenSearch index는 매 실행 시 재생성(`_reset_index`) — idempotent 보장

---

## 6. 디렉토리 구조

```
hackathon/
├── main.py                  # run_pipeline() — 파이프라인 진입점
├── runner.py                # 외부(streamlit) import용 wrapper
├── config.py                # 환경 변수 / 모델 ID / dimension
├── .env                     # AWS / OpenSearch / 모델 비밀
│
├── agents/
│   └── data_analysis.py     # Data Analyst Agent (Profiling/Strategy/Refine/Sanity)
│
├── opensearch/
│   ├── default_index.py     # V1 인덱스 매핑
│   ├── default_search.py    # V1 검색 (TEXT match + knn)
│   ├── agentic_index.py     # V2/V3 인덱스 매핑 (+nori, DCR)
│   ├── agentic_search.py    # search_v2 / search_v3
│   └── ingest.py            # V1/V2/V3 별 bulk ingest
│
├── tools/
│   ├── file_reader.py       # PDF/Excel/CSV 로더 + LLM PDF QA 추출기
│   ├── preprocessing.py     # chunking + V1/V2/V3 preprocess
│   ├── embedding.py         # Titan v2 호출 (LRU 캐시)
│   ├── keyword_extraction.py# LLM 키워드 추출
│   ├── hyde.py              # HyDE 본문 합성
│   ├── add_dcr.py           # DCR 사전 자동 생성
│   ├── analyzer.py          # nori_custom analyzer 빌더
│   └── mcp_tools.py         # Agentic tool dispatch
│
├── evaluation/
│   ├── evaluator.py         # QA 평가 (top-1 substring)
│   └── error_analysis.py    # 실패 패턴 분류
│
├── prompts/                 # LLM 프롬프트 (4종)
└── data/                    # 데모 데이터 (.txt + .xlsx)

streamlit/
└── main.py                  # UI — 업로드/실행/시각화
```

---

## 7. 시연 시나리오

1. 사용자가 streamlit UI에서 `v123_demo_data.txt` + `v123_demo_qa.xlsx` 업로드
2. "RUN PIPELINE" 클릭 → `run_pipeline()` 호출
3. STEP 1: Data Analyst Agent가 데이터를 보고 strategy JSON + decision_log 출력
4. STEP 2~4: V1 → V2 → V3 순차 빌드, 각 단계 점수 출력
5. Streamlit이 막대그래프로 V1<V2<V3 점진 향상을 시각화
6. (점수 미달 시) Strategy Refinement → 재시도, 그래도 부진하면 Data Sanity 진단

---

## 8. 향후 확장 방향

- **벡터 DB 다양화**: pgvector, Pinecone, Weaviate 어댑터 추가
- **임베딩 모델 비교**: Titan v2 vs Cohere v3 vs Bedrock의 새 모델 자동 벤치마크
- **인덱스 스키마 자동 학습**: 사용자 도메인별 최적 매핑을 epoch별로 학습
- **다국어**: 일본어 (sudachi) / 중국어 (smartcn) analyzer 자동 선택
