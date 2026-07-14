"""Shared factories for the seedproof test suite.

Everything is deterministic and offline: records are built in memory, token
ids derive from a stable hash of the token text (so the same text always
gets the same id across factory calls, like a real tokenizer), and files
only ever live under pytest's ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

# Allow running the suite from a clean checkout without `pip install -e .`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seedproof import RunConfig, RunRecord, Token, TokenChoice, save  # noqa: E402

DEFAULT_PROMPT = "Why is the sky blue?"


def stable_id(text: str) -> int:
    """Same text -> same id, like a real tokenizer's vocab lookup."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:6], 16)


def make_tokens(
    texts: Sequence[str],
    with_ids: bool = True,
    logprobs: Optional[Sequence[Optional[float]]] = None,
    tops: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> List[Token]:
    """Build a token stream; ``tops`` maps position -> [(text, logprob), ...]."""
    tokens = []
    for index, text in enumerate(texts):
        top = []
        for choice_text, choice_lp in (tops or {}).get(index, []):
            top.append(
                TokenChoice(
                    text=choice_text,
                    logprob=choice_lp,
                    id=stable_id(choice_text) if with_ids else None,
                )
            )
        tokens.append(
            Token(
                text=text,
                id=stable_id(text) if with_ids else None,
                logprob=logprobs[index] if logprobs else None,
                top=top,
            )
        )
    return tokens


def make_record(
    texts: Sequence[str],
    name: str = "run",
    prompt: str = DEFAULT_PROMPT,
    with_ids: bool = True,
    logprobs: Optional[Sequence[Optional[float]]] = None,
    tops: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    **config: object,
) -> RunRecord:
    """One-line record factory; ``config`` kwargs override RunConfig fields."""
    return RunRecord(
        prompt=prompt,
        config=RunConfig(**config),  # type: ignore[arg-type]
        tokens=make_tokens(texts, with_ids=with_ids, logprobs=logprobs, tops=tops),
        name=name,
    )


@pytest.fixture
def write_record(tmp_path):
    """Persist a record under tmp_path and return its path."""

    def _write(record: RunRecord, filename: Optional[str] = None) -> str:
        path = str(tmp_path / (filename or f"{record.name}.json"))
        save(record, path)
        return path

    return _write
