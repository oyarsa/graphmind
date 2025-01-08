"""Use the OpenAI GPT models to process paper data."""

from paper.gpt.classify_contexts import (
    PaperWithContextClassfied as PaperWithContextClassfied,
)
from paper.gpt.model import Paper as Paper
from paper.gpt.model import PaperAnnotated as PaperAnnotated
from paper.gpt.model import PaperTerms as PaperTerms
from paper.gpt.model import PeerReadAnnotated as PeerReadAnnotated
from paper.gpt.run_gpt import PromptResult as PromptResult
