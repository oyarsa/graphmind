base := "uv run gpt eval reviews run" + \
        " --peerread ../output/peerread_merged_50_111.json" + \
        " --mode binary -n0 -m gpt-4o-mini" + \
        " --output tmp/reviews/" + datetime("%FT%T")

# Print highlighted message
_t msg:
    @echo '>>> {{ style("warning") }}{{msg}}{{ NORMAL }}'

# Original demonstrations file (one demo per rating)
review-og: (_t "Review evaluation with original demonstrations")
    {{base}} --demos review_5

# One demo per rating, but the rationales are edited to be novelty-specific
review-clean: (_t "Review evaluation with clean demonstrations")
    {{base}} --demos review_clean_5

# Clean demos with extraction - simple
extraction-simple: (
    _t "Review evaluation with rationale extraction - simple prompt with extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt simple --extract-demo-prompt abstract

# Clean demos with extraction - basic
extraction-basic: (
    _t "Review evaluation with rationale extraction - basic prompt with extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt basic --extract-demo-prompt abstract

# Clean demos with extraction - overall without extraction demos
extraction-overall-nodemo: (
    _t "Review evaluation with rationale extraction - overall prompt without extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt overall

# Clean demos with extraction - overall with extraction demos
extraction-overall: (
    _t "Review evaluation with rationale extraction - overall prompt with extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt overall --extract-demo-prompt abstract

# Clean demos with full dataset (all 183 entries)
review-full: (_t "Review evaluation with clean demonstrations")
    {{base}} --demos review_clean_5 --peerread ../output/peerread_merged.json
