base := "uv run gpt eval reviews run" + \
        " --peerread ../output/peerread_merged_50_111.json" + \
        " --mode binary -n0 -m gpt-4o-mini" + \
        " --output tmp/reviews/" + `date -Is`

# Original demonstrations file (one demo per rating)
review-og:
    {{base}} --demos review_5

# One demo per rating, but the rationales are edited to be novelty-specific
review-clean:
    {{base}} --demos review_clean_5

# Clean demos with extraction - simple
extraction-simple:
    {{base}} --demos review_clean_5 --extract-prompt simple

# Clean demos with extraction - basic
extraction-basic:
    {{base}} --demos review_clean_5 --extract-prompt basic

# Clean demos with extraction - overall without extraction demos
extraction-overall-nodemo:
    {{base}} --demos review_clean_5 --extract-prompt overall

# Clean demos with extraction - overall with extraction demos
extraction-overall:
    {{base}} --demos review_clean_5 --extract-prompt overall --extract-demo-prompt abstract
