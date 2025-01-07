# SciMON: Scientific Inspiration Machines Optimized for Novelty

Paper: https://arxiv.org/pdf/2305.14259

SciMON uses three graphs (citation, semantic and KG) to retrieve inspiration terms that
are used by the LLM to generate novel scientific ideas.

This module is a reproduction of this method using the same principles but different
methods. In partcular, SciMON uses a pipeline of models to extract the key terms and
context from the paper abstracts, whereas we use GPT prompting to do the same.

## More information

For full documentation on how to use the tools, see [`CLI.md`](./CLI.md).
