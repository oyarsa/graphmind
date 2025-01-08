"""Combine multiple JSON files (each an object) into a single JSON file (array).

Usage:
    python combine_json_files.py <file1> <file2> ...
"""

import json
import sys
from typing import Any

files = sys.argv[1:]

output: list[dict[str, Any]] = []

for file in files:
    with open(file) as f:
        output.append(json.load(f))

print(json.dumps(output))
