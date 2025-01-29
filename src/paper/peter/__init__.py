"""PETER: Paper Evaluation Through Entity Relations.

Use paper graphs to evaluate a paper.
"""

from paper.peter.citations import PaperWithContextClassfied as PaperWithContextClassfied
from paper.peter.graph import Graph as Graph
from paper.peter.graph import graph_from_json as graph_from_json
from paper.peter.semantic import PaperAnnotated as PaperAnnotated
