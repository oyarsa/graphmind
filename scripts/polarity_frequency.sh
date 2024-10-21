#!/bin/sh

if [ $# -ne 1 ]; then
	echo "Usage: $0 <filename>"
	exit 1
fi

jq -r '.[].references[].contexts_annotated[].polarity' "$1" | sort | uniq -c

printf "\nTotal: %s\n" \
	"$(jq -r '[.[].references[].contexts_annotated[].polarity] | length' "$1")"
