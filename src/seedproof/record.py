"""Canonical run-record model: one recorded generation, one JSON file.

A *run record* captures everything seedproof needs to compare two
generations after the fact: the prompt, the runtime configuration (backend,
quantization, seed, sampler settings), and the emitted token stream with
optional per-token log-probabilities and top-k candidates.

The on-disk format is versioned JSON with sorted keys and a trailing newline,
written atomically, so records diff cleanly in git and never end up
half-written. The full schema is documented in ``docs/record-format.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import RecordError

#: Version of the on-disk record format. Bump when a field changes meaning.
RECORD_VERSION = 1

#: Config fields that control *which* token the sampler picks.
SAMPLING_FIELDS: Tuple[str, ...] = ("sampler", "temperature", "top_k", "top_p")

#: Config fields that describe *where* the math ran.
RUNTIME_FIELDS: Tuple[str, ...] = ("backend", "device", "quant")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordError(message)


@dataclass
class TokenChoice:
    """One candidate in a token's top-k list."""

    text: str
    logprob: float
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"text": self.text, "logprob": self.logprob}
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: Any, where: str) -> "TokenChoice":
        _require(isinstance(data, dict), f"{where}: top entry must be an object")
        _require(isinstance(data.get("text"), str), f"{where}: top entry needs a string 'text'")
        logprob = data.get("logprob")
        _require(
            isinstance(logprob, (int, float)) and not isinstance(logprob, bool),
            f"{where}: top entry needs a numeric 'logprob'",
        )
        token_id = data.get("id")
        _require(
            token_id is None or (isinstance(token_id, int) and not isinstance(token_id, bool)),
            f"{where}: top entry 'id' must be an integer",
        )
        return cls(text=data["text"], logprob=float(logprob), id=token_id)


@dataclass
class Token:
    """One emitted token: text always, id / logprob / top-k when captured."""

    text: str
    id: Optional[int] = None
    logprob: Optional[float] = None
    top: List[TokenChoice] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"text": self.text}
        if self.id is not None:
            data["id"] = self.id
        if self.logprob is not None:
            data["logprob"] = self.logprob
        if self.top:
            data["top"] = [choice.to_dict() for choice in self.top]
        return data

    @classmethod
    def from_dict(cls, data: Any, index: int) -> "Token":
        where = f"token[{index}]"
        _require(isinstance(data, dict), f"{where}: must be an object")
        _require(isinstance(data.get("text"), str), f"{where}: needs a string 'text'")
        token_id = data.get("id")
        _require(
            token_id is None or (isinstance(token_id, int) and not isinstance(token_id, bool)),
            f"{where}: 'id' must be an integer",
        )
        logprob = data.get("logprob")
        _require(
            logprob is None
            or (isinstance(logprob, (int, float)) and not isinstance(logprob, bool)),
            f"{where}: 'logprob' must be a number",
        )
        top_raw = data.get("top", [])
        _require(isinstance(top_raw, list), f"{where}: 'top' must be a list")
        top = [TokenChoice.from_dict(entry, where) for entry in top_raw]
        return cls(
            text=data["text"],
            id=token_id,
            logprob=None if logprob is None else float(logprob),
            top=top,
        )


@dataclass
class RunConfig:
    """The runtime configuration a generation ran under.

    ``extra`` is a free-form string-keyed map for runtime knobs seedproof
    does not model explicitly (batch size, thread count, kernel flags);
    entries are compared verbatim and treated as runtime axes.
    """

    model: str = ""
    backend: str = ""
    device: str = ""
    quant: str = ""
    seed: Optional[int] = None
    sampler: str = "greedy"
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_stochastic(self) -> bool:
        """True when the sampler can pick anything but the argmax token."""
        return self.sampler != "greedy" and self.temperature > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "device": self.device,
            "quant": self.quant,
            "seed": self.seed,
            "sampler": self.sampler,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RunConfig":
        _require(isinstance(data, dict), "config: must be an object")
        known = {
            "model", "backend", "device", "quant", "seed",
            "sampler", "temperature", "top_k", "top_p", "extra",
        }
        for key in data:
            _require(key in known, f"config: unknown field '{key}' (use 'extra' for custom knobs)")
        for key in ("model", "backend", "device", "quant", "sampler"):
            value = data.get(key, "")
            _require(isinstance(value, str), f"config: '{key}' must be a string")
        seed = data.get("seed")
        _require(
            seed is None or (isinstance(seed, int) and not isinstance(seed, bool)),
            "config: 'seed' must be an integer",
        )
        for key in ("temperature", "top_p"):
            value = data.get(key, 0.0)
            _require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"config: '{key}' must be a number",
            )
        top_k = data.get("top_k", 0)
        _require(
            isinstance(top_k, int) and not isinstance(top_k, bool),
            "config: 'top_k' must be an integer",
        )
        extra = data.get("extra", {})
        _require(isinstance(extra, dict), "config: 'extra' must be an object")
        for key in extra:
            _require(isinstance(key, str), "config: 'extra' keys must be strings")
        return cls(
            model=data.get("model", ""),
            backend=data.get("backend", ""),
            device=data.get("device", ""),
            quant=data.get("quant", ""),
            seed=seed,
            sampler=data.get("sampler", "greedy"),
            temperature=float(data.get("temperature", 0.0)),
            top_k=top_k,
            top_p=float(data.get("top_p", 1.0)),
            extra=dict(extra),
        )

    def fingerprint(self) -> str:
        """Stable 12-hex-digit digest of the whole configuration."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return _sha256(canonical)[:12]

    def summary(self) -> str:
        """Compact one-line description used in reports."""
        parts = []
        if self.backend or self.device:
            hw = self.backend + (f":{self.device}" if self.device else "")
            parts.append(hw)
        if self.quant:
            parts.append(self.quant)
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        sampler = self.sampler
        if self.temperature or sampler != "greedy":
            sampler += f"@t={self.temperature:g}"
        parts.append(sampler)
        return "  ".join(parts)


def compare_configs(a: RunConfig, b: RunConfig) -> Dict[str, Tuple[Any, Any]]:
    """Return the config fields where ``a`` and ``b`` differ.

    Keys are field names; ``extra`` entries are flattened to ``extra.<key>``.
    The dict preserves a stable, documented order: named fields first, then
    extra keys sorted alphabetically.
    """
    deltas: Dict[str, Tuple[Any, Any]] = {}
    for name in ("model", "backend", "device", "quant", "seed", *SAMPLING_FIELDS):
        va, vb = getattr(a, name), getattr(b, name)
        if va != vb:
            deltas[name] = (va, vb)
    for key in sorted(set(a.extra) | set(b.extra)):
        va, vb = a.extra.get(key), b.extra.get(key)
        if va != vb:
            deltas[f"extra.{key}"] = (va, vb)
    return deltas


@dataclass
class RunRecord:
    """One recorded generation: prompt + config + token stream."""

    prompt: str
    config: RunConfig
    tokens: List[Token]
    name: str = ""
    created_at: str = ""
    notes: str = ""

    # -- derived views ------------------------------------------------------

    def texts(self) -> List[str]:
        return [token.text for token in self.tokens]

    def ids(self) -> Optional[List[int]]:
        """All token ids, or ``None`` when any token was captured without one."""
        ids: List[int] = []
        for token in self.tokens:
            if token.id is None:
                return None
            ids.append(token.id)
        return ids

    def decoded(self) -> str:
        """The generation as plain text (token texts concatenated)."""
        return "".join(self.texts())

    def has_logprobs(self) -> bool:
        return bool(self.tokens) and all(t.logprob is not None for t in self.tokens)

    def prompt_sha256(self) -> str:
        return _sha256(self.prompt)

    def stream_fingerprint(self, basis: str = "text") -> str:
        """12-hex-digit digest of the token stream on the given basis."""
        if basis == "id":
            ids = self.ids()
            if ids is None:
                raise RecordError(
                    f"record '{self.name}': cannot fingerprint by id — "
                    "some tokens were captured without ids"
                )
            payload = "id:" + ",".join(str(i) for i in ids)
        elif basis == "text":
            payload = "text:" + "\x1f".join(self.texts())
        else:
            raise RecordError(f"unknown fingerprint basis '{basis}' (use 'id' or 'text')")
        return _sha256(payload)[:12]

    # -- (de)serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "seedproof_record": RECORD_VERSION,
            "name": self.name,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256(),
            "config": self.config.to_dict(),
            "tokens": [token.to_dict() for token in self.tokens],
        }
        if self.created_at:
            data["created_at"] = self.created_at
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "RunRecord":
        _require(isinstance(data, dict), "record: top level must be a JSON object")
        version = data.get("seedproof_record")
        _require(
            version is not None,
            "record: missing 'seedproof_record' version key — not a seedproof record",
        )
        _require(
            isinstance(version, int) and not isinstance(version, bool),
            "record: 'seedproof_record' must be an integer",
        )
        _require(
            version <= RECORD_VERSION,
            f"record: format version {version} is newer than this tool "
            f"(supports up to {RECORD_VERSION}) — upgrade seedproof",
        )
        prompt = data.get("prompt")
        _require(isinstance(prompt, str), "record: needs a string 'prompt'")
        for key in ("name", "created_at", "notes"):
            _require(isinstance(data.get(key, ""), str), f"record: '{key}' must be a string")
        tokens_raw = data.get("tokens")
        _require(isinstance(tokens_raw, list), "record: needs a 'tokens' list")
        tokens = [Token.from_dict(entry, i) for i, entry in enumerate(tokens_raw)]
        record = cls(
            prompt=prompt,
            config=RunConfig.from_dict(data.get("config", {})),
            tokens=tokens,
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
        )
        stored_hash = data.get("prompt_sha256")
        if stored_hash is not None:
            _require(isinstance(stored_hash, str), "record: 'prompt_sha256' must be a string")
            _require(
                stored_hash == record.prompt_sha256(),
                "record: prompt_sha256 does not match the prompt — "
                "the record was edited or corrupted after capture",
            )
        return record


def dumps(record: RunRecord) -> str:
    """Serialize a record to canonical JSON (sorted keys, trailing newline)."""
    return json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def loads(text: str, name: str = "") -> RunRecord:
    """Parse a record from a JSON string; ``name`` fills in a missing label."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RecordError(f"record: invalid JSON — {exc}") from None
    record = RunRecord.from_dict(data)
    if not record.name and name:
        record.name = name
    return record


def load(path: str) -> RunRecord:
    """Load a record file; an unnamed record is labelled after its filename."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return loads(text, name=stem)
    except RecordError as exc:
        raise RecordError(f"{path}: {exc}") from None


def save(record: RunRecord, path: str) -> None:
    """Write a record atomically: temp file in the same directory + rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(dumps(record))
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
