# Current Work: Empty Evidence Bug Fix

## Quick Start
```bash
just api-dev  # Start backend on port 8000
```

## Problem
The Novelty Assessment section was showing empty `supporting_evidence` and `contradictory_evidence` arrays, even though related papers were being found.

---

## Completed Fixes

### 1. Template placeholders not substituted (MAIN BUG)
**File:** `src/paper/gpt/prompts/summarise_related_peter.py`
- Template used `{{title_main}}` (double braces) instead of `{title_main}`
- Python's `.format()` escapes `{{` to literal `{`, so variables weren't substituted
- LLM received literal `{title_related}` placeholders instead of actual paper titles
- This caused hallucinated generic summaries for all papers

### 2. Missing paper IDs in prompt
**File:** `src/paper/gpt/evaluate_paper_graph.py`
- `format_related` now includes `paper_id` in the prompt
- LLM can now reference papers by their IDs in evidence

### 3. Missing paper source in prompt
**File:** `src/paper/gpt/evaluate_paper_graph.py`
- `format_related` now includes `source` (citations/semantic) in the prompt
- LLM can correctly attribute evidence to citation vs semantic papers

### 4. Date filtering too strict
**File:** `src/paper/single_paper/related_papers.py`
- Changed `year < main_year` to `year <= main_year`
- Now includes papers from the same year as the main paper

### 5. Only querying "recent" pool from S2 API
**File:** `src/paper/single_paper/paper_retrieval.py`
- Now queries both "recent" AND "all-cs" pools from Semantic Scholar
- Unions results before filtering, giving access to older papers
- Before: 28 papers all from 2025-2026 (all filtered out for 2024 paper)
- After: 50 papers with years 2018-2026, 19 pass the date filter

---

## Status
- All fixes implemented and committed
- `just lint` passes (354 tests passed, exit code 247 is pyright version warning, not error)

## Still TODO
1. **Verify semantic papers appear in evidence** - Need to run full test and check that evidence includes both `citations` and `semantic` sources
2. **Test with Playwright on frontend** - Verify end-to-end functionality

## Test Commands
```bash
# Check evidence sources (takes ~2 min due to API calls)
curl -s -N "http://localhost:8000/mind/evaluate?id=2406.18245v2&title=Weak%20Reward%20Model%20Transforms%20Generative%20Models%20into%20Robust%20Causal%20Event%20Extraction%20Systems&filter_by_date=true&llm_model=gpt-4o-mini" | grep "^data:" | tail -1 | sed 's/^data: //' | jq '{
  related_sources: [.result.result.related[].source] | group_by(.) | map({(.[0]): length}) | add,
  evidence_sources: [.result.result.paper.structured_evaluation.supporting_evidence[].source] | group_by(.) | map({(.[0]): length}) | add
}'
```

Expected result: `evidence_sources` should include both `citations` and `semantic`.

## Files Modified
- `src/paper/gpt/prompts/summarise_related_peter.py` - Fixed template braces
- `src/paper/gpt/evaluate_paper_graph.py` - Added paper_id and source to format_related
- `src/paper/single_paper/related_papers.py` - Fixed date filter (< to <=)
- `src/paper/single_paper/paper_retrieval.py` - Query both S2 pools
