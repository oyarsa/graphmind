"""Remove "empty" (no real data) S2 results from semantic_scholar_best.json

As a result of `merge_s2_api_results.py`, which generates semantic_scholar_best.json
some items have just a title and no other attributes. These have not matches, so we're
removing it from the final/best output.
"""

import json
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile) as f:
    data = json.load(f)

output = [d for d in data if "title" in d]

print(len(data), "before")
print(len(output), "after")

with open(outfile, "w") as f:
    json.dump(output, f, indent=2)
