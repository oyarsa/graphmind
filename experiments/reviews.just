cmd := "uv run gpt eval reviews run --mode binary -n0 -m gpt-4o-mini" + \
        " --output tmp/reviews/" + datetime("%FT%T")
base := cmd + " --peerread ../output/peerread_merged_50_111.json"
full := cmd + " --peerread ../output/peerread_merged.json"

# Print highlighted message
_t msg:
    @echo '{{ style("warning") }}>>> {{msg}}{{ NORMAL }}'

# Original demonstrations file (one demo per rating)
[group: 'review-only']
review-og: (_t "Review evaluation with original demonstrations")
    {{base}} --demos review_5

# One demo per rating, but the rationales are edited to be novelty-specific
[group: 'review-only']
review-clean: (_t "Review evaluation with clean demonstrations")
    {{base}} --demos review_clean_5

# Clean demos with extraction - simple
[group: 'extraction']
extraction-simple: (
    _t "Review evaluation with rationale extraction - simple prompt with extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt simple --extract-demo-prompt abstract

# Clean demos with extraction - basic
[group: 'extraction']
extraction-basic: (
    _t "Review evaluation with rationale extraction - basic prompt with extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt basic --extract-demo-prompt abstract

# Clean demos with extraction - overall without extraction demos
[group: 'extraction']
extraction-overall-nodemo: (
    _t "Review evaluation with rationale extraction - overall prompt without extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt overall

# Clean demos with extraction - overall with extraction demos
[group: 'extraction']
extraction-overall: (
    _t "Review evaluation with rationale extraction - overall prompt with extraction demos"
)
    {{base}} --demos review_clean_5 --extract-prompt overall --extract-demo-prompt abstract

# All base runs
all-base:
    @just review-og
    @just review-clean
    @just extraction-simple
    @just extraction-basic
    @just extraction-overall-nodemo
    @just extraction-overall

# Original demonstrations file (one demo per rating) - full data
[group: 'review-only']
review-og-full: (_t "Review evaluation with original demonstrations - full data")
    {{base}} --demos review_5

# One demo per rating, but the rationales are edited to be novelty-specific - full data
[group: 'review-only']
review-clean-full: (_t "Review evaluation with clean demonstrations - full data")
    {{base}} --demos review_clean_5

# Clean demos with extraction - simple - full data
[group: 'extraction']
extraction-simple-full: (
    _t "Review evaluation with rationale extraction - simple prompt with extraction demos - full data"
)
    {{base}} --demos review_clean_5 --extract-prompt simple --extract-demo-prompt abstract

# Clean demos with extraction - basic - full data
[group: 'extraction']
extraction-basic-full: (
    _t "Review evaluation with rationale extraction - basic prompt with extraction demos - full data"
)
    {{base}} --demos review_clean_5 --extract-prompt basic --extract-demo-prompt abstract

# Clean demos with extraction - overall without extraction demos - full data
[group: 'extraction']
extraction-overall-nodemo-full: (
    _t "Review evaluation with rationale extraction - overall prompt without extraction demos - full data"
)
    {{base}} --demos review_clean_5 --extract-prompt overall

# Clean demos with extraction - overall with extraction demos - full data
[group: 'extraction']
extraction-overall-full: (
    _t "Review evaluation with rationale extraction - overall prompt with extraction demos - full data"
)
    {{base}} --demos review_clean_5 --extract-prompt overall --extract-demo-prompt abstract

# All full runs
all-full:
    @just review-og-full
    @just review-clean-full
    @just extraction-simple-full
    @just extraction-basic-full
    @just extraction-overall-nodemo-full
    @just extraction-overall-full
