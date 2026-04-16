"""Shared MC answer-sheet layout constants and text-wrapping helpers.

Both the generation layer (page packing / row sizing) and the PDF
renderer consume these values.  Keeping them in one place prevents
the two modules from diverging on layout arithmetic.
"""

from __future__ import annotations

import re
import textwrap

PROMPT_LINE_GAP = 28
PROMPT_LINE_SPACING = 14
PROMPT_WRAP_WIDTH = 52
CHOICE_LEGEND_TOP_OFFSET = 4
CHOICE_LEGEND_LINE_SPACING = 14
CHOICE_WRAP_WIDTH = 46


def wrap_prompt_text(text: str) -> list[str]:
    wrapped = textwrap.wrap(
        text,
        width=PROMPT_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [text]


def wrap_choice_text(bubble_label: str, text: str) -> list[str]:
    wrapped = textwrap.wrap(
        f"{bubble_label}. {text}",
        width=CHOICE_WRAP_WIDTH,
        subsequent_indent="   ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [f"{bubble_label}. {text}"]


def display_prompt(question_number: int, prompt: str) -> str:
    normalized_prompt = re.sub(
        rf"^\s*question\s+{question_number}\s*[:.\-]\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    return f"{question_number}. {normalized_prompt}"
