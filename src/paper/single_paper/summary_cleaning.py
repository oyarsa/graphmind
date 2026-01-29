"""Post-processing utilities for cleaning LLM-generated evidence summaries."""

import re

# Patterns that indicate formulaic openings we want to remove.
# These get stripped and the remainder is capitalised.
_FORMULAIC_PREFIXES = re.compile(
    r"^(the\s+)?"
    r"(related|supporting|contrasting|second|first|target|main|prior)\s+"
    r"paper"
    r"(,\s*[^,]+,)?"  # Optional appositive like ", 'Title Here',"
    r"\s+"
    r"(also\s+)?"
    r"(demonstrates?|shows?|highlights?|emphasi[sz]es?|provides?|presents?|"
    r"introduces?|focuses\s+on|supports?|contrasts?(\s+with(\s+the\s+first)?)?"
    r"|corroborates?|reinforces?|validates?|confirms?|aligns?\s+with|"
    r"diverges?\s+from|differs?\s+from)\s+"
    r"(that\s+|by\s+)?",  # Optional "that" or "by" after verb
    re.IGNORECASE,
)

# Patterns like "The findings from the supporting paper show that..."
# or "The findings from the supporting paper highlight the importance..."
_FINDINGS_PREFIX = re.compile(
    r"^the\s+findings\s+(from|of|in)\s+the\s+"
    r"(related|supporting|contrasting|second|first|target|main|prior)\s+"
    r"paper\s+"
    r"(demonstrate|show|highlight|emphasi[sz]e|indicate|suggest|reveal)\s+"
    r"(that\s+)?",
    re.IGNORECASE,
)

# Pattern like "The development of X in the supporting paper provides..."
_DEVELOPMENT_PREFIX = re.compile(
    r"^the\s+(development|introduction|use|application)\s+of\s+[^.]+?\s+in\s+the\s+"
    r"(related|supporting|contrasting|second|first|target|main|prior)\s+"
    r"paper\s+"
    r"(provides?|demonstrates?|shows?|offers?|enables?)\s+",
    re.IGNORECASE,
)


def clean_summary(text: str) -> str:
    """Remove formulaic openings from evidence summaries.

    Strips phrases like "The supporting paper demonstrates that..." or
    "The findings from the contrasting paper show that..." and capitalises
    the remainder.

    Args:
        text: Raw summary text from the LLM.

    Returns:
        Cleaned summary with formulaic prefix removed.
    """
    # Try patterns from most specific to least specific
    for pattern in (_DEVELOPMENT_PREFIX, _FINDINGS_PREFIX, _FORMULAIC_PREFIXES):
        cleaned = pattern.sub("", text)
        if cleaned != text:
            return cleaned[0].upper() + cleaned[1:] if cleaned else text

    return text
