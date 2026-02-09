# API Server

## Starting the Server

```bash
just api-dev
```

Server runs at http://127.0.0.1:8000 with docs at http://127.0.0.1:8000/docs

## Testing SSE Endpoints

```bash
# Call endpoint and save response (shows progress spinner)
uv run python scripts/call_sse.py "http://127.0.0.1:8000/mind/evaluate?id=2406.18245v2&title=..." /tmp/out.json

# Inspect specific fields
cat /tmp/out.json | jq '.result.result.paper.structured_evaluation | {label, supporting: [.supporting_evidence[].source], contradictory: [.contradictory_evidence[].source]}'
```

## Common Test Cases

1. Evidence distribution (3 semantic + fill with citations):
   ```bash
   uv run python scripts/call_sse.py \
     "http://127.0.0.1:8000/mind/evaluate?id=2406.18245v2&title=Weak%20Reward%20Model&llm_model=gpt-4o-mini" \
     /tmp/evidence.json
   cat /tmp/evidence.json | jq '{
     supporting: [.result.result.paper.structured_evaluation.supporting_evidence[].source],
     contradictory: [.result.result.paper.structured_evaluation.contradictory_evidence[].source]
   }'
   # Expected: up to 3 "semantic" followed by "citations" (max 5 total per array)
   ```

2. Basic evaluation check:
   ```bash
   cat /tmp/out.json | jq '.result.result.paper.structured_evaluation | {label, has_summary: (.paper_summary | length > 0)}'
   ```

## Response Structure

For `/mind/evaluate` and `/mind/evaluate-abstract`:

- `result.result.paper.structured_evaluation.label` - Novelty score (1-5)
- `result.result.paper.structured_evaluation.paper_summary` - Summary of contributions
- `result.result.paper.structured_evaluation.supporting_evidence[]` - Evidence items with `.source`, `.text`, `.paper_title`
- `result.result.paper.structured_evaluation.contradictory_evidence[]` - Same structure
- `result.result.paper.structured_evaluation.key_comparisons[]` - Technical comparisons
- `result.result.paper.structured_evaluation.conclusion` - Final assessment
