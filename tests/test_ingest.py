"""Ingest adapters: generic JSON, JSONL token events, captured SSE streams."""

from __future__ import annotations

import json
import os

import pytest

from seedproof import IngestError, RunConfig, ingest


# -- generic ------------------------------------------------------------------


def test_generic_accepts_bare_strings_and_synonym_keys():
    bare = ingest(json.dumps({"tokens": ["Hel", "lo"]}), "generic")
    assert bare.texts() == ["Hel", "lo"]
    assert bare.ids() is None
    payload = {
        "tokens": [
            {"token": "Hi", "token_id": 5, "log_prob": -0.25,
             "top_logprobs": [{"token": "Hi", "log_prob": -0.25}]},
        ]
    }
    token = ingest(json.dumps(payload), "generic").tokens[0]
    assert token.text == "Hi"
    assert token.id == 5
    assert token.logprob == -0.25
    assert token.top[0].text == "Hi"


def test_generic_extracts_metadata_and_explicit_flags_win():
    payload = {
        "prompt": "Why?",
        "model": "local-8b",
        "config": {"backend": "cpu", "quant": "q8_0"},
        "tokens": ["x"],
    }
    record = ingest(json.dumps(payload), "generic")
    assert record.prompt == "Why?"
    assert record.config.model == "local-8b"
    assert record.config.backend == "cpu"
    assert record.config.quant == "q8_0"
    # The capture said cpu, but the operator knows it ran on cuda:
    forced = ingest(json.dumps(payload), "generic",
                    config=RunConfig(backend="cuda", seed=7))
    assert forced.config.backend == "cuda"
    assert forced.config.seed == 7
    assert forced.config.quant == "q8_0"  # non-overridden values survive


def test_generic_structural_errors_are_specific():
    with pytest.raises(IngestError, match="'tokens'"):
        ingest(json.dumps({"text": "no tokens here"}), "generic")
    with pytest.raises(IngestError, match="invalid JSON"):
        ingest("{oops", "generic")
    with pytest.raises(IngestError, match=r"tokens\[1\]"):
        ingest(json.dumps({"tokens": ["ok", 42]}), "generic")


def test_value_errors_bool_logprob_empty_capture_unknown_format():
    # bool subclasses int; `"logprob": true` is a capture bug, not a number.
    with pytest.raises(IngestError, match="logprob"):
        ingest(json.dumps({"tokens": [{"text": "x", "logprob": True}]}), "generic")
    with pytest.raises(IngestError, match="no tokens"):
        ingest(json.dumps({"tokens": []}), "generic")
    with pytest.raises(IngestError, match="unknown format"):
        ingest("{}", "csv")


# -- jsonl --------------------------------------------------------------------


def test_jsonl_parses_one_token_per_line_skipping_blanks():
    text = '{"text": "a", "id": 1}\n\n{"text": "b", "id": 2}\n'
    record = ingest(text, "jsonl")
    assert record.texts() == ["a", "b"]
    assert record.ids() == [1, 2]


def test_jsonl_reports_the_failing_line_number():
    text = '{"text": "a"}\nnot json\n'
    with pytest.raises(IngestError, match="line 2"):
        ingest(text, "jsonl")


# -- sse ----------------------------------------------------------------------


def _chunk(content, logprob=None, top=None, model="local-8b-q4"):
    choice = {"index": 0, "delta": {"content": content}, "finish_reason": None}
    if logprob is not None:
        entry = {"token": content, "logprob": logprob}
        if top is not None:
            entry["top_logprobs"] = [
                {"token": t, "logprob": lp} for t, lp in top
            ]
        choice["logprobs"] = {"content": [entry]}
    return "data: " + json.dumps({"model": model, "choices": [choice]})


def test_sse_extracts_tokens_logprobs_top_k_and_model():
    lines = [
        _chunk("The", -0.05, top=[("The", -0.05), ("A", -3.1)]),
        _chunk(" sky", -0.12, top=[(" sky", -0.12), (" sun", -2.2)]),
        "data: [DONE]",
    ]
    record = ingest("\r\n".join(lines) + "\r\n", "sse")  # CRLF like real curl
    assert record.texts() == ["The", " sky"]
    assert record.tokens[0].logprob == -0.05
    assert record.tokens[1].top[1].text == " sun"
    assert record.config.model == "local-8b-q4"


def test_sse_respects_done_and_skips_non_token_lines():
    role_only = json.dumps(
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    )
    finish = json.dumps(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    lines = [
        ": a comment line",
        "event: message",
        f"data: {role_only}",
        _chunk("kept"),
        f"data: {finish}",
        "",
        "data: [DONE]",
        _chunk("ignored"),  # anything after [DONE] must not be read
    ]
    record = ingest("\n".join(lines), "sse")
    assert record.texts() == ["kept"]


def test_sse_legacy_text_field_and_batched_logprob_chunks():
    # Legacy /v1/completions puts the delta in choices[0].text:
    legacy = "data: " + json.dumps({"choices": [{"index": 0, "text": "Hello"}]})
    assert ingest(legacy + "\ndata: [DONE]", "sse").texts() == ["Hello"]
    # Some servers batch several logprob entries into one chunk:
    entries = [
        {"token": "a", "logprob": -0.1},
        {"token": "b", "logprob": -0.2},
    ]
    payload = {"choices": [{"index": 0, "delta": {"content": "ab"},
                            "logprobs": {"content": entries}}]}
    record = ingest("data: " + json.dumps(payload), "sse")
    assert record.texts() == ["a", "b"]


def test_sse_bad_payload_reports_the_line_number():
    with pytest.raises(IngestError, match="line 2"):
        ingest(_chunk("ok") + "\ndata: {broken\n", "sse")


def test_sse_capture_example_file_ingests():
    # The shipped example capture is the contract for the README quickstart.
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "sse-capture.txt")
    with open(path, encoding="utf-8") as handle:
        record = ingest(handle.read(), "sse", prompt="Why is the sky blue?")
    assert record.texts() == ["The", " sky", " appears", " blue", "."]
    assert record.has_logprobs()
    assert record.config.model == "local-8b-q4"
    assert all(len(token.top) == 2 for token in record.tokens)
