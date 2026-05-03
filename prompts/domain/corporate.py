CORPORATE_DOMAIN_HINT = """\
Data domain: Korean corporate policy and approval authority document.
Typical content includes:
  - Welfare regulations (경조사, 교통비, 복지 지원 한도 등)
  - Approval authority tables (전결권자, 금액 구간별 승인자)
  - Conditional clauses and exception rules (특례 조항, 날짜 기준 분기)
  - Structured tables (금액 구간 / 전결권자 매핑)
Documents may contain intentional or unintentional data inconsistencies (Dirty Data).
"""

CORPORATE_FEW_SHOT = """\
## Reference Examples

--- Example 1: Unstructured Policy Document (복지규정.txt) ---

Profile observed:
  language.ko_ratio = 0.95
  structure.type = "semi-structured"
  structure.has_consistent_schema = false
  content.semantic_complexity = "high"   (조건부 예외 조항, 날짜 기준 분기)
  content.keyword_density = "low"        (자연어 문장 중심)
  domain.type = "legal"
  domain.specificity = "medium"

Expected strategy:
  use_analyzer: true    (ko_ratio >= 0.5)
  use_keyword:  false   (keyword_density low, specificity medium)
  use_hyde:     true    (semantic_complexity high + domain legal)
  use_hybrid:   false   (use_keyword false)
  index_version: "v3"   (use_analyzer + use_hyde both true)
  chunking_strategy: "sentence"

--- Example 2: Structured Approval Table (전결권자리스트.xlsx) ---

Profile observed:
  language.ko_ratio = 0.85
  structure.type = "structured"
  structure.has_consistent_schema = true
  content.keyword_density = "high"       (구분코드, 전결권자명 반복)
  content.semantic_complexity = "low"    (테이블 구조, 단순 매핑)
  domain.type = "legal"
  domain.specificity = "high"            (전결, 실장, 팀장 등 도메인 용어)

Expected strategy:
  use_analyzer: true    (ko_ratio >= 0.5)
  use_keyword:  true    (keyword_density high + specificity high)
  use_hyde:     false   (semantic_complexity low)
  use_hybrid:   true    (use_keyword true)
  index_version: "v2"   (use_keyword true, use_hyde false)
  chunking_strategy: "structure-aware"
"""
