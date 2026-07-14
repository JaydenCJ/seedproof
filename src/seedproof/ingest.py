"""Ingest adapters: turn captured runtime output into canonical run records.

seedproof never talks to a model server itself — you capture what your
runtime emitted (a curl of a streaming endpoint, a JSONL token log, a JSON
dump) and ingest it. Three formats are supported in 0.1.0:

- ``generic`` — a JSON object with a ``tokens`` list; each token is either a
  bare string or an object. Key synonyms are accepted (``text``/``token``/
  ``content``, ``id``/``token_id``, ``logprob``/``log_prob``,
  ``top``/``top_logprobs``) so most ad-hoc dumps ingest unchanged.
- ``jsonl``  — one token event per line, same token shape as ``generic``.
- ``sse``    — a captured Server-Sent-Events stream from an
  OpenAI-compatible ``chat/completions`` (or legacy ``completions``)
  endpoint: ``data: {...}`` lines, terminated by ``data: [DONE]``. Per-token
  logprobs and ``top_logprobs`` are extracted when the capture has them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .errors import IngestError
from .record import RunConfig, RunRecord, Token, TokenChoice

#: Formats accepted by :func:`ingest` and the CLI ``--format`` flag.
FORMATS: Tuple[str, ...] = ("generic", "jsonl", "sse")

_TEXT_KEYS = ("text", "token", "content")
_ID_KEYS = ("id", "token_id")
_LOGPROB_KEYS = ("logprob", "log_prob")
_TOP_KEYS = ("top", "top_logprobs")


def ingest(
    text: str,
    fmt: str,
    config: Optional[RunConfig] = None,
    prompt: str = "",
    name: str = "",
    notes: str = "",
) -> RunRecord:
    """Parse a capture in the given format and return a validated record."""
    if fmt == "generic":
        tokens, found = _parse_generic(text)
    elif fmt == "jsonl":
        tokens, found = _parse_jsonl(text)
    elif fmt == "sse":
        tokens, found = _parse_sse(text)
    else:
        raise IngestError(f"unknown format '{fmt}' (supported: {', '.join(FORMATS)})")
    if not tokens:
        raise IngestError(f"{fmt}: capture contains no tokens")
    merged = _merge_config(config, found)
    return RunRecord(
        prompt=prompt or found.get("prompt", ""),
        config=merged,
        tokens=tokens,
        name=name,
        notes=notes,
    )


def _merge_config(explicit: Optional[RunConfig], found: Dict[str, Any]) -> RunConfig:
    """Explicit config wins; capture-provided values fill the gaps."""
    base = found.get("config")
    merged = RunConfig.from_dict(base.to_dict() if isinstance(base, RunConfig) else (base or {}))
    if "model" in found and not merged.model:
        merged.model = found["model"]
    if explicit is None:
        return merged
    defaults = RunConfig()
    for field_name in ("model", "backend", "device", "quant", "seed",
                       "sampler", "temperature", "top_k", "top_p"):
        value = getattr(explicit, field_name)
        if value != getattr(defaults, field_name):
            setattr(merged, field_name, value)
    merged.extra.update(explicit.extra)
    return merged


# -- token coercion -----------------------------------------------------------


def _pick(data: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _coerce_choice(data: Any, where: str) -> TokenChoice:
    if not isinstance(data, dict):
        raise IngestError(f"{where}: top entry must be an object, got {type(data).__name__}")
    text = _pick(data, _TEXT_KEYS)
    if not isinstance(text, str):
        raise IngestError(f"{where}: top entry needs a string text/token field")
    logprob = _pick(data, _LOGPROB_KEYS)
    if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
        raise IngestError(f"{where}: top entry needs a numeric logprob")
    token_id = _pick(data, _ID_KEYS)
    if token_id is not None and (not isinstance(token_id, int) or isinstance(token_id, bool)):
        raise IngestError(f"{where}: top entry id must be an integer")
    return TokenChoice(text=text, logprob=float(logprob), id=token_id)


def _coerce_token(data: Any, where: str) -> Token:
    """Accept a bare string or a dict with synonym keys; reject the rest."""
    if isinstance(data, str):
        return Token(text=data)
    if not isinstance(data, dict):
        raise IngestError(
            f"{where}: token must be a string or an object, got {type(data).__name__}"
        )
    text = _pick(data, _TEXT_KEYS)
    if not isinstance(text, str):
        raise IngestError(f"{where}: token needs a string text/token/content field")
    token_id = _pick(data, _ID_KEYS)
    if token_id is not None and (not isinstance(token_id, int) or isinstance(token_id, bool)):
        raise IngestError(f"{where}: token id must be an integer")
    logprob = _pick(data, _LOGPROB_KEYS)
    if logprob is not None and (not isinstance(logprob, (int, float)) or isinstance(logprob, bool)):
        raise IngestError(f"{where}: token logprob must be a number")
    top_raw = _pick(data, _TOP_KEYS) or []
    if not isinstance(top_raw, list):
        raise IngestError(f"{where}: top/top_logprobs must be a list")
    top = [_coerce_choice(entry, where) for entry in top_raw]
    return Token(
        text=text,
        id=token_id,
        logprob=None if logprob is None else float(logprob),
        top=top,
    )


# -- generic ------------------------------------------------------------------


def _parse_generic(text: str) -> Tuple[List[Token], Dict[str, Any]]:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise IngestError(f"generic: invalid JSON — {exc}") from None
    if not isinstance(data, dict):
        raise IngestError("generic: top level must be a JSON object with a 'tokens' list")
    tokens_raw = data.get("tokens")
    if not isinstance(tokens_raw, list):
        raise IngestError("generic: missing 'tokens' list")
    tokens = [_coerce_token(entry, f"tokens[{i}]") for i, entry in enumerate(tokens_raw)]
    found: Dict[str, Any] = {}
    if isinstance(data.get("prompt"), str):
        found["prompt"] = data["prompt"]
    if isinstance(data.get("model"), str):
        found["model"] = data["model"]
    if isinstance(data.get("config"), dict):
        found["config"] = data["config"]
    return tokens, found


# -- jsonl --------------------------------------------------------------------


def _parse_jsonl(text: str) -> Tuple[List[Token], Dict[str, Any]]:
    tokens: List[Token] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError as exc:
            raise IngestError(f"jsonl: line {line_no}: invalid JSON — {exc}") from None
        tokens.append(_coerce_token(data, f"line {line_no}"))
    return tokens, {}


# -- sse ----------------------------------------------------------------------


def _parse_sse(text: str) -> Tuple[List[Token], Dict[str, Any]]:
    """Parse a captured OpenAI-compatible streaming response.

    Only ``data:`` lines carry payloads; ``event:``/``id:`` lines, comment
    lines (``:``), and blank event separators are skipped per the SSE spec.
    Limitation (documented): multi-line ``data:`` fields are not reassembled —
    completion chunks are single-line JSON in practice.
    """
    tokens: List[Token] = []
    found: Dict[str, Any] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError as exc:
            raise IngestError(f"sse: line {line_no}: invalid JSON payload — {exc}") from None
        if not isinstance(chunk, dict):
            raise IngestError(f"sse: line {line_no}: payload must be a JSON object")
        if isinstance(chunk.get("model"), str) and "model" not in found:
            found["model"] = chunk["model"]
        tokens.extend(_tokens_from_chunk(chunk, line_no))
    return tokens, found


def _tokens_from_chunk(chunk: Dict[str, Any], line_no: int) -> List[Token]:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, dict):
        raise IngestError(f"sse: line {line_no}: choices[0] must be an object")
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
        return [
            _token_from_logprob_entry(entry, line_no, i)
            for i, entry in enumerate(logprobs["content"])
        ]
    # No logprobs: fall back to the raw text delta.
    delta = choice.get("delta")
    text: Any = None
    if isinstance(delta, dict):
        text = delta.get("content")
    if text is None:
        text = choice.get("text")  # legacy completions endpoint
    if text is None or text == "":
        return []  # role-only first chunk, finish chunk, etc.
    if not isinstance(text, str):
        raise IngestError(f"sse: line {line_no}: delta content must be a string")
    return [Token(text=text)]


def _token_from_logprob_entry(entry: Any, line_no: int, index: int) -> Token:
    where = f"sse: line {line_no}: logprobs.content[{index}]"
    if not isinstance(entry, dict):
        raise IngestError(f"{where}: must be an object")
    text = entry.get("token")
    if not isinstance(text, str):
        raise IngestError(f"{where}: needs a string 'token'")
    logprob = entry.get("logprob")
    if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
        raise IngestError(f"{where}: needs a numeric 'logprob'")
    top_raw = entry.get("top_logprobs") or []
    if not isinstance(top_raw, list):
        raise IngestError(f"{where}: top_logprobs must be a list")
    top = [_coerce_choice(item, where) for item in top_raw]
    return Token(text=text, logprob=float(logprob), top=top)
