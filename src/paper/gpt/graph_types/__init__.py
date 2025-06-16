"""Organise multiple implementations of Graph output types."""

from collections.abc import Mapping, Sequence

from paper.gpt.graph_types.base import GPTGraphBase
from paper.gpt.graph_types.excerpts import GPTExcerpt
from paper.gpt.graph_types.full import GPTGraph
from paper.gpt.graph_types.nodetail import GPTGraphNoDetail
from paper.gpt.graph_types.noexperiments import GPTGraphNoExperiments
from paper.gpt.graph_types.nomethods import GPTGraphNoMethods

_TYPES: Mapping[str, type[GPTGraphBase]] = {
    "full": GPTGraph,
    "nodetail": GPTGraphNoDetail,
    "noexperiments": GPTGraphNoExperiments,
    "nomethods": GPTGraphNoMethods,
    "excerpts": GPTExcerpt,
}
VALID_TYPES: Sequence[str] = sorted(_TYPES)


def get_graph_type(type_name: str) -> type[GPTGraphBase]:
    """Get output type from type name.

    You can use this function to retrieve the appropriate type based on the
    `GraphPrompt.type_name`.

    Raises:
        ValueError: if the type name is invalid. See `VALID_TYPES`.
    """
    if type_name not in _TYPES:
        raise ValueError(
            f"Invalid graph type: '{type_name}'. Must be one of: {VALID_TYPES}"
        )
    return _TYPES[type_name]
