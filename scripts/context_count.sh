#!/bin/sh

if [ $# -ne 1 ]; then
	echo "Usage: $0 <filename>"
	exit 1
fi

jq '[.[].references[].contexts[]] | length' "$1"
