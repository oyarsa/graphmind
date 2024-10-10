"""Show all unique values for the ratings.

These are usually "number - text", so we need to look at the possibilities to decide
how to parse them.
"""

import json
import sys

file = sys.argv[1]

with open(file) as f:
    data = json.load(f)

ratings: set[str] = set()

for item in data:
    for rating in item["ratings_text"]:
        ratings.add(rating)

print("\n".join(sorted(ratings)))
