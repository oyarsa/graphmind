#!/usr/bin/env fish

# Construct data subset, annotate main and related papers and build graph.

set datadir output/subset-50
set limit 0

echo ">>>>>>> CONSTRUCT DATASET"
uv run src/paper/construct_dataset.py \
    --references output/semantic_scholar_final.json \
    --recommended output/recomendations/papers_recommended.json \
    --asap output/asap_balanced_50.json \
    --output $datadir \
    --min-refs 4

echo
echo ">>>>>>> ASAP TERMS"
uv run gpt terms run $datadir/asap_with_s2_references.json $datadir/asap_terms \
    --clean-run -n $limit \
    --prompt-term multi \
    --prompt-abstract simple \
    --abstract-demos src/paper/gpt/prompts/abstract_demonstrations_10.json \
    --paper-type asap

echo
echo ">>>>>>> RELATED TERMS"
uv run gpt terms run $datadir/asap_related.json $datadir/related_terms \
    --clean-run -n $limit \
    --prompt-term multi \
    --prompt-abstract simple \
    --abstract-demos src/paper/gpt/prompts/abstract_demonstrations_10.json \
    --paper-type s2

echo
echo ">>>>>> BUILD GRAPHS"
uv run scimon build \
    --ann $datadir/related_terms/results_valid.json \
    --asap $datadir/asap_with_s2_references.json \
    --output $datadir/graphs.json
