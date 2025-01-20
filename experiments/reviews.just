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

# Clean demonstrations with novelty rationale extraction before rating - simple prompt
extraction-simple:
    {{base}} --demos review_clean_5 --extract-prompt simple

# Clean demonstrations with novelty rationale extraction before rating - basic prompt
extraction-basic:
    {{base}} --demos review_clean_5 --extract-prompt basic
