RETAIL_DOMAIN_HINT = """\
Data domain: Korean retail product recommendation dataset.
Typical structure per record:
  카테고리 / 상품명 / 가격 / 평점 / 해시태그 / 추천상황
Records are short, structured, and contain domain-specific hashtag keywords.
"""

RETAIL_FEW_SHOT = """\
## Reference Example (Retail Data)

Profile observed:
  language.ko_ratio = 0.85
  content.keyword_density = "high"   (hashtag fields present)
  structure.has_consistent_schema = true
  domain.specificity = "high"        (fashion/beauty/food jargon)
  content.semantic_complexity = "low" (structured product records)

Expected strategy output:
  use_analyzer: true   (ko_ratio >= 0.5)
  use_keyword:  true   (keyword_density high + domain specificity high)
  use_hyde:     false  (semantic_complexity low)
  use_hybrid:   true   (use_keyword true)
  index_version: "v2"  (use_keyword true, use_hyde false)
  chunking_strategy: "structure-aware" (has_consistent_schema true)
"""
