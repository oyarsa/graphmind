#!/bin/sh
uv run src/paper/semantic_scholar/info.py main \
	output/asap_filtered.json tmp/info_main -n 10

uv run src/paper/semantic_scholar/info.py references \
	output/asap_filtered.json tmp/info_references -n 2

uv run src/paper/semantic_scholar/recommended.py \
	tmp/info_main/final.json tmp/info_recommended \
	--limit-recommendations 20

uv run src/paper/construct_dataset.py \
	--asap output/asap_filtered.json \
	--references tmp/info_references/final.json \
	--recommended tmp/info_recommended/papers_recommended.json \
	--output tmp/subset
