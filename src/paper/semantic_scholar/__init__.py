"""Use the Semantic Scholar API to retrieve data about papers."""

from paper.semantic_scholar.model import ASAPWithFullS2 as ASAPWithFullS2
from paper.semantic_scholar.model import Author as Author
from paper.semantic_scholar.model import Paper as Paper
from paper.semantic_scholar.model import PaperArea as PaperArea
from paper.semantic_scholar.model import PaperFromASAP as PaperFromASAP
from paper.semantic_scholar.model import (
    PaperRecommended as PaperRecommended,
)
from paper.semantic_scholar.model import PaperWithS2Refs as PaperWithS2Refs
from paper.semantic_scholar.model import S2Reference as S2Reference
from paper.semantic_scholar.model import Tldr as Tldr
from paper.semantic_scholar.model import clean_title as clean_title
