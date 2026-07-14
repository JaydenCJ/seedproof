"""Token-stream alignment: find the first divergent token and measure drift.

This module is pure math over two :class:`~seedproof.record.RunRecord`
streams — no I/O, no config interpretation (that lives in ``forensics``).
It answers four questions:

1. Where do the streams first diverge, and on what basis (id vs text)?
2. Do the streams reconverge afterwards (a transient flip) or fully derail?
3. Was the divergence a *near tie* — did the losing token trail the winner
   by less than epsilon, the signature of numerical noise?
4. Was numerical drift already accumulating along the shared prefix,
   before any token flipped?
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .errors import AlignError
from .record import RunRecord, Token

#: Log-probability gap (nats) at or under which two candidates count as tied.
#: fp16-vs-fp32 kernel noise typically lands well below this.
DEFAULT_TIE_EPSILON = 0.05

#: Minimum run of matching tokens after a divergence to call it reconverged.
RESYNC_MIN_TOKENS = 3


@dataclass
class Divergence:
    """Where two token streams part ways.

    ``kind`` is one of:

    - ``identical`` — same length, every position matches; ``index`` is None.
    - ``token``     — position ``index`` holds different tokens.
    - ``length``    — one stream is a strict prefix of the other; ``index``
      is the length of the shorter stream and exactly one of
      ``a_token``/``b_token`` is set (the first unmatched token).
    """

    kind: str
    basis: str
    index: Optional[int]
    common_len: int
    a_token: Optional[Token] = None
    b_token: Optional[Token] = None

    @property
    def diverged(self) -> bool:
        return self.kind != "identical"


@dataclass
class Resync:
    """Streams realigned after the divergence: a run of matching tokens."""

    a_index: int
    b_index: int
    length: int


@dataclass
class PrefixDrift:
    """Log-probability disagreement along the *shared* prefix.

    Even before any token flips, two runtimes disagree numerically; the
    per-token |Δlogprob| along the matching prefix makes that visible.
    ``trend`` is ``accumulating`` when the second half of the prefix drifts
    at more than twice the rate of the first half, ``flat`` otherwise, and
    ``none`` when neither record carried logprobs.
    """

    count: int
    max_abs_delta: float = 0.0
    mean_abs_delta: float = 0.0
    max_index: int = -1
    trend: str = "none"


@dataclass
class TieAnalysis:
    """Log-probability evidence at the divergent position.

    ``cross_gap`` is how far the *other* run's chosen token trailed the
    winner, measured inside one run's own top-k list (minimum over both
    directions). A tiny cross gap means the two candidates were effectively
    tied and any numerical wobble could flip the argmax.
    """

    available: bool
    cross_gap: Optional[float] = None
    runner_up_gap: Optional[float] = None
    near_tie: bool = False
    absent_from_top: bool = False


def choose_basis(a: RunRecord, b: RunRecord, requested: str = "auto") -> str:
    """Pick the comparison basis: token ids when both runs captured them.

    Ids are the ground truth — two tokenizers can render different ids as the
    same text. ``auto`` falls back to text when either run lacks full ids;
    requesting ``id`` explicitly in that situation is an error.
    """
    if requested == "text":
        return "text"
    has_ids = a.ids() is not None and b.ids() is not None
    if requested == "id":
        if not has_ids:
            raise AlignError(
                "cannot compare by id: at least one record has tokens without ids "
                "(re-capture with ids, or pass --basis text)"
            )
        return "id"
    if requested == "auto":
        return "id" if has_ids else "text"
    raise AlignError(f"unknown basis '{requested}' (use auto, id, or text)")


def _keys(record: RunRecord, basis: str) -> List[object]:
    if basis == "id":
        ids = record.ids()
        assert ids is not None  # guarded by choose_basis
        return list(ids)
    return list(record.texts())


def first_divergence(a: RunRecord, b: RunRecord, basis: str = "auto") -> Divergence:
    """Locate the first position where the two streams differ."""
    resolved = choose_basis(a, b, basis)
    keys_a, keys_b = _keys(a, resolved), _keys(b, resolved)
    common = min(len(keys_a), len(keys_b))
    for index in range(common):
        if keys_a[index] != keys_b[index]:
            return Divergence(
                kind="token",
                basis=resolved,
                index=index,
                common_len=index,
                a_token=a.tokens[index],
                b_token=b.tokens[index],
            )
    if len(keys_a) == len(keys_b):
        return Divergence(kind="identical", basis=resolved, index=None, common_len=common)
    longer_a = len(keys_a) > len(keys_b)
    return Divergence(
        kind="length",
        basis=resolved,
        index=common,
        common_len=common,
        a_token=a.tokens[common] if longer_a else None,
        b_token=b.tokens[common] if not longer_a else None,
    )


def find_resync(
    a: RunRecord,
    b: RunRecord,
    start: int,
    min_len: int = RESYNC_MIN_TOKENS,
) -> Optional[Resync]:
    """Find the earliest run of >= ``min_len`` matching tokens after ``start``.

    A resync means the flip was transient (both continuations found their way
    back to the same text); no resync means the streams fully derailed.
    Matching is by text — after a divergence the two runs are different
    sentences, and text is what a human checks realignment against.
    """
    tail_a, tail_b = a.texts()[start:], b.texts()[start:]
    matcher = difflib.SequenceMatcher(a=tail_a, b=tail_b, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size >= min_len:
            return Resync(
                a_index=start + block.a,
                b_index=start + block.b,
                length=block.size,
            )
    return None


def prefix_drift(a: RunRecord, b: RunRecord, common_len: int) -> PrefixDrift:
    """Measure |Δlogprob| per token over the shared prefix."""
    deltas: List[Tuple[int, float]] = []
    for index in range(common_len):
        lp_a, lp_b = a.tokens[index].logprob, b.tokens[index].logprob
        if lp_a is None or lp_b is None:
            continue
        deltas.append((index, abs(lp_a - lp_b)))
    if not deltas:
        return PrefixDrift(count=0)
    values = [delta for _, delta in deltas]
    max_value = max(values)
    max_index = next(index for index, delta in deltas if delta == max_value)
    trend = "flat"
    if len(values) >= 8:
        half = len(values) // 2
        first_mean = sum(values[:half]) / half
        second_mean = sum(values[half:]) / (len(values) - half)
        if second_mean > 2.0 * first_mean and second_mean > 1e-9:
            trend = "accumulating"
    return PrefixDrift(
        count=len(values),
        max_abs_delta=max_value,
        mean_abs_delta=sum(values) / len(values),
        max_index=max_index,
        trend=trend,
    )


def _winner_logprob(token: Token, basis: str) -> Optional[float]:
    """Logprob of the token that actually won, from the token or its top list."""
    if token.logprob is not None:
        return token.logprob
    for choice in token.top:
        if _choice_matches(choice, token, basis):
            return choice.logprob
    return None


def _choice_matches(choice: object, token: Token, basis: str) -> bool:
    if basis == "id" and getattr(choice, "id", None) is not None and token.id is not None:
        return choice.id == token.id  # type: ignore[attr-defined]
    return choice.text == token.text  # type: ignore[attr-defined]


def _cross_gap(winner: Token, rival: Token, basis: str) -> Tuple[Optional[float], bool]:
    """Gap between ``winner``'s choice and ``rival``'s choice in winner's top-k.

    Returns ``(gap, rival_absent)``. ``gap`` is None when the evidence is not
    there (no top-k captured, or the rival token never made the list).
    """
    if not winner.top:
        return None, False
    winner_lp = _winner_logprob(winner, basis)
    if winner_lp is None:
        return None, False
    for choice in winner.top:
        if _choice_matches(choice, rival, basis):
            return winner_lp - choice.logprob, False
    return None, True


def analyze_tie(
    a: RunRecord,
    b: RunRecord,
    divergence: Divergence,
    epsilon: float = DEFAULT_TIE_EPSILON,
) -> TieAnalysis:
    """Was the divergence a numerical tie-break? Needs top-k in the records."""
    if divergence.kind != "token" or divergence.index is None:
        return TieAnalysis(available=False)
    token_a, token_b = a.tokens[divergence.index], b.tokens[divergence.index]
    gap_ab, absent_ab = _cross_gap(token_a, token_b, divergence.basis)
    gap_ba, absent_ba = _cross_gap(token_b, token_a, divergence.basis)
    gaps = [gap for gap in (gap_ab, gap_ba) if gap is not None]
    runner_up = _runner_up_gap(token_a, divergence.basis)
    if not gaps and runner_up is None:
        return TieAnalysis(available=False, absent_from_top=absent_ab or absent_ba)
    # Minimum by magnitude: either run's list may witness the near tie.
    cross = min(gaps, key=abs) if gaps else None
    near = cross is not None and abs(cross) <= epsilon
    return TieAnalysis(
        available=True,
        cross_gap=cross,
        runner_up_gap=runner_up,
        near_tie=near,
        absent_from_top=absent_ab or absent_ba,
    )


def _runner_up_gap(token: Token, basis: str) -> Optional[float]:
    """Winner-vs-best-loser gap inside one token's own top-k list."""
    winner_lp = _winner_logprob(token, basis)
    if winner_lp is None or len(token.top) < 2:
        return None
    losers = [c.logprob for c in token.top if not _choice_matches(c, token, basis)]
    if not losers:
        return None
    return winner_lp - max(losers)
