#!/usr/bin/env fish

# Construct data subset, annotate main and related papers, build graph and query papers.

set datadir output/subset-50
set limit 0

echo ">>>>>>> CONSTRUCT DATASET"
uv run src/paper/construct_dataset.py \
    --references output/semantic_scholar_final.json \
    --recommended output/recomendations/papers_recommended.json \
    --peerread output/peerread_balanced_50.json \
    --output $datadir \
    --min-refs 4

echo
echo ">>>>>>> PeerRead TERMS"
uv run gpt terms run $datadir/peerread_with_s2_references.json $datadir/peerread_terms \
    --clean-run -n $limit \
    --prompt-term multi \
    --prompt-abstract simple \
    --abstract-demos src/paper/gpt/prompts/abstract_demonstrations_10.json \
    --paper-type peerread

echo
echo ">>>>>>> RELATED TERMS"
uv run gpt terms run $datadir/peerread_related.json $datadir/related_terms \
    --clean-run -n $limit \
    --prompt-term multi \
    --prompt-abstract simple \
    --abstract-demos src/paper/gpt/prompts/abstract_demonstrations_10.json \
    --paper-type s2

echo
echo ">>>>>> BUILD GRAPHS"
uv run scimon build \
    --ann $datadir/related_terms/results_valid.json \
    --peerread $datadir/peerread_with_s2_references.json \
    --output $datadir/graphs.json

echo
echo ">>>>>> GRAPH QUERY ANNOTATED"
uv run scimon query-peerread \
    --ann-peerread $datadir/peerread_terms/results_valid.json \
    --graph $datadir/graphs.json \
    --output $datadir/peerread_with_graph.json
