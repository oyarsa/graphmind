"""Show paper data (title, abstract and introduction) along with the prompt.

Works as a demonstration to what we're sending to the LLM.
"""

import json
import sys
from pathlib import Path


def show(item: dict[str, str]) -> str:
    lines = [
        "Title: " + item["title"],
        "\n",
        "Abstract: " + item["abstract"],
        "\n",
        "Introduction: " + item["introduction"],
    ]
    return "\n".join(lines)


def prompt(item: dict[str, str]) -> str:
    return (
        "Given the following paper title, abstract and introduction, what are the"
        " key concepts from the paper? Respond with one concept per line.\n\n"
        f"{show(item)}\n\n"
    )


data = json.loads(Path(sys.argv[1]).read_text())
item = data[0]
print(prompt(item))
