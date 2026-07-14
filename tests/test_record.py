"""Record model: (de)serialization, validation, fingerprints, atomic writes."""

from __future__ import annotations

import json
import os

import pytest

from conftest import make_record
from seedproof import (
    RECORD_VERSION,
    RecordError,
    RunConfig,
    compare_configs,
    dumps,
    load,
    loads,
    save,
)


def test_save_then_load_roundtrips_every_field(tmp_path):
    record = make_record(
        ["Hello", " world"],
        name="roundtrip",
        backend="cuda",
        quant="q4_k_m",
        seed=42,
        logprobs=[-0.1, -0.2],
        tops={0: [("Hello", -0.1), ("Hi", -2.0)]},
    )
    record.config.extra = {"threads": 8}
    path = str(tmp_path / "r.json")
    save(record, path)
    loaded = load(path)
    assert loaded.texts() == ["Hello", " world"]
    assert loaded.ids() == record.ids()
    assert loaded.tokens[0].logprob == -0.1
    assert loaded.tokens[0].top[1].text == "Hi"
    assert loaded.config.backend == "cuda"
    assert loaded.config.extra == {"threads": 8}
    assert loaded.prompt == record.prompt


def test_dumps_is_canonical_and_unicode_safe(tmp_path):
    text = dumps(make_record([" 空", "は青い"], name="jp"))
    assert text.endswith("\n")
    data = json.loads(text)
    assert list(data) == sorted(data)  # sorted keys => clean git diffs
    assert " 空" in text  # ensure_ascii=False keeps CJK readable
    path = str(tmp_path / "jp.json")
    save(make_record([" 空", "は青い"], name="jp"), path)
    assert load(path).decoded() == " 空は青い"


def test_load_names_record_from_filename_unless_explicit(tmp_path):
    unnamed = str(tmp_path / "gpu-run.json")
    save(make_record(["x"], name=""), unnamed)
    assert load(unnamed).name == "gpu-run"
    named = str(tmp_path / "whatever.json")
    save(make_record(["x"], name="explicit"), named)
    assert load(named).name == "explicit"


def test_tampered_prompt_hash_is_rejected():
    # A record whose prompt was edited after capture must fail loudly:
    # silent acceptance would poison every downstream comparison.
    data = make_record(["x"], prompt="original").to_dict()
    data["prompt"] = "edited"
    with pytest.raises(RecordError, match="prompt_sha256"):
        loads(json.dumps(data))


def test_version_key_and_json_validation(tmp_path):
    # Not a record at all:
    with pytest.raises(RecordError, match="seedproof_record"):
        loads(json.dumps({"prompt": "p", "tokens": []}))
    # A future format must ask for an upgrade instead of misreading fields:
    data = make_record(["x"]).to_dict()
    data["seedproof_record"] = RECORD_VERSION + 1
    with pytest.raises(RecordError, match="upgrade"):
        loads(json.dumps(data))
    # Broken JSON errors carry the file path for actionable messages:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RecordError, match="broken.json"):
        load(str(path))


def test_token_without_text_reports_its_index():
    data = make_record(["a", "b"]).to_dict()
    del data["tokens"][1]["text"]
    with pytest.raises(RecordError, match=r"token\[1\]"):
        loads(json.dumps(data))


def test_config_validation_rejects_unknown_fields_and_bool_seed():
    # A typo like "quanto" must not become an invisible, uncompared field.
    with pytest.raises(RecordError, match="quanto"):
        RunConfig.from_dict({"quanto": "q4"})
    # bool subclasses int in Python; a capture bug writing `true` must not
    # silently become seed=1.
    with pytest.raises(RecordError, match="seed"):
        RunConfig.from_dict({"seed": True})


def test_ids_view_and_stream_fingerprints():
    record = make_record(["a", "b"])
    text_fp = record.stream_fingerprint("text")
    id_fp = record.stream_fingerprint("id")
    assert len(text_fp) == len(id_fp) == 12
    assert text_fp != id_fp
    assert record.stream_fingerprint("text") == text_fp  # deterministic
    record.tokens[1].id = None
    assert record.ids() is None
    with pytest.raises(RecordError, match="without ids"):
        record.stream_fingerprint("id")


def test_is_stochastic_requires_both_sampler_and_temperature():
    assert RunConfig(sampler="top-p", temperature=0.8).is_stochastic()
    assert not RunConfig(sampler="greedy", temperature=0.8).is_stochastic()
    assert not RunConfig(sampler="top-p", temperature=0.0).is_stochastic()


def test_compare_configs_flattens_extra_keys():
    a = RunConfig(backend="cpu", extra={"threads": 4})
    b = RunConfig(backend="cuda", extra={"threads": 8, "batch": 2})
    deltas = compare_configs(a, b)
    assert deltas["backend"] == ("cpu", "cuda")
    assert deltas["extra.threads"] == (4, 8)
    assert deltas["extra.batch"] == (None, 2)


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = str(tmp_path / "sub" / "r.json")
    save(make_record(["a"]), path)
    assert os.path.exists(path)
    # Only the record itself lives in the directory: no orphaned temp files.
    assert os.listdir(os.path.dirname(path)) == ["r.json"]
