"""Chinese text preprocessing utilities for the Novel-to-Script Adaptation pipeline.

Every function in this module is a pure function with no side effects and no
file I/O.  The module provides mechanical text cleanup operations:

- ``unify_quotes`` — convert CJK corner-bracket quotes to standard quotes
- ``normalize_paragraphs`` — normalise line endings and blank-line runs
- ``strip_bom`` — remove a UTF-8 BOM if present
- ``preprocess`` — run the full pipeline and return a ``PreprocessArtifact``
"""

import re

from engine.models import PreprocessArtifact


def unify_quotes(text: str) -> str:
    """Convert all Chinese corner-bracket quote pairs to standard quotes.

    * ``「」`` (CJK corner brackets, U+300C / U+300D) → ``""``
    * ``『』`` (CJK white corner brackets, U+300E / U+300F) → ``''``
    * Already-standard ``""`` and ``''`` are left untouched.

    Parameters
    ----------
    text : str
        Input text that may contain CJK corner-bracket quotes.

    Returns
    -------
    str
        Text with corner-bracket quotes replaced by standard equivalents.
    """
    result = text.replace("「", '"').replace("」", '"')
    result = result.replace("『", "'").replace("』", "'")
    return result


def normalize_paragraphs(text: str) -> str:
    """Normalise paragraph line-endings and blank-line runs.

    1. Convert ``\\r\\n`` → ``\\n`` and standalone ``\\r`` → ``\\n``.
    2. Strip trailing whitespace from each line.
    3. Collapse three or more consecutive ``\\n`` characters into exactly two
       (i.e. a single blank-line paragraph separator).

    Parameters
    ----------
    text : str
        Raw text with potentially mixed line endings and indentation.

    Returns
    -------
    str
        Text with normalised line endings and paragraph spacing.
    """
    # Step 1: normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Step 2: strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Step 3: collapse 3+ consecutive newlines → exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def strip_bom(text: str) -> str:
    """Remove a UTF-8 BOM (``\\ufeff``) from the start of *text* if present.

    Parameters
    ----------
    text : str
        Input string that may begin with a BOM.

    Returns
    -------
    str
        The string without a leading BOM character.
    """
    if text.startswith("﻿"):
        return text[1:]
    return text


def preprocess(text: str, source_path: str) -> PreprocessArtifact:
    """Run the full preprocessing pipeline and wrap the result in an artifact.

    Pipeline order::

        strip_bom → unify_quotes → normalize_paragraphs

    The result is stripped of leading/trailing whitespace before being
    stored in the artifact.

    Parameters
    ----------
    text : str
        Raw novel text.
    source_path : str
        Original file path (for provenance tracking).

    Returns
    -------
    PreprocessArtifact
        Pydantic model with ``cleaned_text``, ``total_chars``,
        ``original_path`` and ``schema_version`` fields.
    """
    cleaned = strip_bom(text)
    cleaned = unify_quotes(cleaned)
    cleaned = normalize_paragraphs(cleaned)
    cleaned = cleaned.strip()

    return PreprocessArtifact(
        schema_version="1.0",
        original_path=source_path,
        cleaned_text=cleaned,
        total_chars=len(cleaned),
    )
