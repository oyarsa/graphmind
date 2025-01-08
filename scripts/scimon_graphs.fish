#!/usr/bin/env fish

# Given extracted terms, build the SciMON graph and use it to query the papers.

set datadir output/subset-50

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
