"""Prompts for summarising related papers in PETER-style format."""

from paper.gpt.prompts import PromptTemplate

POSITIVE = PromptTemplate(
    name="positive",
    template="""\
The following data contains information from a **Main Paper** and a **Related Paper**. \
The Related Paper has a positive relation to the Main Paper. It contains supporting \
information that strengthens the main paper claims.

Your task is to generate a summary that highlights how the Related Paper supports the \
Main Paper. Your summary should be short and concise, comprising a few sentences only.

#####
-Data-
# Main paper

**Title**: {title_main}
**Abstract**: {abstract_main}

# Related paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Output:
""",
)

NEGATIVE = PromptTemplate(
    name="negative",
    template="""\
The following data contains information from a **Main Paper** and a **Related Paper**. \
The Related Paper has a negative relation to the Main Paper. It is used to contrast the \
claims made by the Main Paper.

Your task is to generate a summary that highlights how the Related Paper contrasts the \
Main Paper. Your summary should be short and concise, comprising of a few sentences \
only.

#####
-Data-
# Main paper

**Title**: {title_main}
**Abstract**: {abstract_main}

# Related paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Output:
""",
)

# -----------------------------------------------------------------------------
# V2 prompts: Improved terminology to prevent LLM from echoing "The Related
# Paper..." in summaries. Uses neutral labels ("first paper", "second paper")
# instead of "Main Paper" / "Related Paper".
# -----------------------------------------------------------------------------

POSITIVE_V2 = PromptTemplate(
    name="positive-v2",
    template="""\
Compare the following two papers. The second paper has a supportive relationship to the \
first - it contains findings that strengthen or corroborate the first paper's claims.

Summarise how the second paper supports the first. Be direct and concise (2-3 sentences). \
Focus on the specific supporting evidence or methodological similarities.

IMPORTANT: Start your summary with the actual finding or methodology, not with phrases like \
"The paper", "This paper", "The supporting paper", or "The second paper". Jump straight into \
what the connection is.

#####
-Data-
# Paper being evaluated

**Title**: {title_main}
**Abstract**: {abstract_main}

# Supporting paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Summary:
""",
)

NEGATIVE_V2 = PromptTemplate(
    name="negative-v2",
    template="""\
Compare the following two papers. The second paper has a contrasting relationship to the \
first - it presents different approaches or findings that contrast with the first paper's claims.

Summarise how the second paper contrasts with the first. Be direct and concise (2-3 sentences). \
Focus on the specific differences in approach, methodology, or findings.

IMPORTANT: Start your summary with the actual finding or methodology, not with phrases like \
"The paper", "This paper", "The contrasting paper", or "The second paper". Jump straight into \
what the difference is.

#####
-Data-
# Paper being evaluated

**Title**: {title_main}
**Abstract**: {abstract_main}

# Contrasting paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Summary:
""",
)

PETER_SUMMARISE_USER_PROMPTS = {
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "positive-v2": POSITIVE_V2,
    "negative-v2": NEGATIVE_V2,
}
