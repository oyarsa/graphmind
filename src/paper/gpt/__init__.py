"""Use the OpenAI GPT models to process paper data."""

from paper.gpt.classify_contexts import (
    PaperWithContextClassfied as PaperWithContextClassfied,
)
from paper.gpt.model import ASAPAnnotated as ASAPAnnotated
from paper.gpt.model import Paper as Paper
from paper.gpt.model import PaperAnnotated as PaperAnnotated
from paper.gpt.model import PaperTerms as PaperTerms
from paper.gpt.run_gpt import PromptResult as PromptResult
