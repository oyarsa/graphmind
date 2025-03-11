# PETER: Paper Evaluation Through Entity Relations

PETER is a method to query related papers from a main paper based on their
relationships. There are two kinds of relations: semantic and citation. For each, there
can be positively or negatively related papers.

The retrieved papers are summarised using GPT (see `gpt.summarise_related_peter`), and
then used to predict whether the paper was approved or rejected.
