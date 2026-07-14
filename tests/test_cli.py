"""CLI behavior: subcommands, exit codes, stderr discipline, JSON output."""

from __future__ import annotations

import io
import json

import pytest

import seedproof
from conftest import make_record
from seedproof.cli import EXIT_DIVERGENT, EXIT_ERROR, EXIT_OK, main


def test_version_flag_and_bare_invocation(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"seedproof {seedproof.__version__}"
    # No subcommand prints help and exits with the error code.
    assert main([]) == EXIT_ERROR
    assert "diff" in capsys.readouterr().out


def test_diff_exit_codes_follow_diff_conventions(write_record, capsys):
    path_x1 = write_record(make_record(["x"], name="a"))
    path_x2 = write_record(make_record(["x"], name="b"))
    assert main(["diff", path_x1, path_x2]) == EXIT_OK
    assert "verdict: identical" in capsys.readouterr().out
    path_a = write_record(make_record(["x", "y"], name="c", backend="cpu"))
    path_b = write_record(make_record(["x", "q"], name="d", backend="cuda"))
    assert main(["diff", path_a, path_b]) == EXIT_DIVERGENT
    out = capsys.readouterr().out
    assert "first divergent token: index 1" in out
    assert "verdict: backend-numerics" in out


def test_diff_json_output_is_machine_readable(write_record, capsys):
    path_a = write_record(make_record(["x", "y"], name="a", backend="cpu"))
    path_b = write_record(make_record(["x", "q"], name="b", backend="cuda"))
    assert main(["diff", "--json", path_a, path_b]) == EXIT_DIVERGENT
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "backend-numerics"
    assert data["divergence"]["index"] == 1
    assert data["a"] == "a" and data["b"] == "b"


def test_diff_error_paths_are_one_line_stderr(tmp_path, write_record, capsys):
    assert main(["diff", "/nonexistent/a.json", "/nonexistent/b.json"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("seedproof: error:")
    assert "Traceback" not in err
    # A JSON file that is not a record is rejected with the reason.
    path_a = write_record(make_record(["x"], name="a"))
    other = tmp_path / "notrecord.json"
    other.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    assert main(["diff", path_a, str(other)]) == EXIT_ERROR
    assert "seedproof_record" in capsys.readouterr().err


def test_diff_basis_text_ignores_id_differences(write_record):
    record_a = make_record(["x", "y"], name="a")
    record_b = make_record(["x", "y"], name="b")
    record_b.tokens[1].id = record_a.tokens[1].id + 1
    path_a, path_b = write_record(record_a), write_record(record_b)
    assert main(["diff", "--basis", "text", path_a, path_b]) == EXIT_OK
    assert main(["diff", "--basis", "id", path_a, path_b]) == EXIT_DIVERGENT


def test_diff_tie_epsilon_flag_changes_the_evidence(write_record, capsys):
    tops_a = {1: [("y", -0.5), ("q", -0.6)]}
    tops_b = {1: [("q", -0.6), ("y", -0.5)]}
    path_a = write_record(make_record(
        ["x", "y"], name="a", backend="cpu", logprobs=[-0.1, -0.5], tops=tops_a))
    path_b = write_record(make_record(
        ["x", "q"], name="b", backend="cuda", logprobs=[-0.1, -0.6], tops=tops_b))
    main(["diff", "--tie-epsilon", "0.5", path_a, path_b])
    assert "tie-break" in capsys.readouterr().out
    main(["diff", "--tie-epsilon", "0.01", path_a, path_b])
    assert "distribution itself moved" in capsys.readouterr().out


def test_show_prints_config_and_tokens(write_record, capsys):
    path = write_record(make_record(["Hello", " world"], name="demo",
                                    backend="cpu", quant="q8_0", seed=3))
    assert main(["show", path]) == EXIT_OK
    out = capsys.readouterr().out
    assert "record: demo" in out
    assert "cpu  q8_0  seed=3" in out
    assert '"Hello"' in out


def test_ls_lists_skips_and_rejects(tmp_path, write_record, capsys):
    write_record(make_record(["x"], name="good", backend="cpu"))
    (tmp_path / "junk.json").write_text("{}", encoding="utf-8")
    assert main(["ls", str(tmp_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "good.json" in out
    assert "skipped" in out  # junk is reported, not fatal
    # ls on a file (not a directory) is a usage error.
    assert main(["ls", str(tmp_path / "good.json")]) == EXIT_ERROR
    assert "not a directory" in capsys.readouterr().err
    # An empty directory is fine and says so.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["ls", str(empty)]) == EXIT_OK
    assert "no records found" in capsys.readouterr().out


def test_matrix_over_directory_exits_zero_even_on_split(tmp_path, write_record, capsys):
    write_record(make_record(["x", "y"], name="c1", backend="cpu"))
    write_record(make_record(["x", "y"], name="c2", backend="cpu"))
    write_record(make_record(["x", "q"], name="g1", backend="cuda"))
    # matrix is a report, not a gate: it must not fail the build by itself.
    assert main(["matrix", str(tmp_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "classes: 2" in out
    assert "explains the split" in out


def test_matrix_json_output(tmp_path, write_record, capsys):
    write_record(make_record(["x", "y"], name="c1", backend="cpu"))
    write_record(make_record(["x", "q"], name="g1", backend="cuda"))
    assert main(["matrix", "--json", str(tmp_path)]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["consistent"] is False
    assert len(data["classes"]) == 2


def test_matrix_mixed_prompts_exit_2(tmp_path, write_record, capsys):
    write_record(make_record(["x"], name="p1", prompt="one"))
    write_record(make_record(["x"], name="p2", prompt="two"))
    assert main(["matrix", str(tmp_path)]) == EXIT_ERROR
    assert "different prompts" in capsys.readouterr().err


def test_check_gate_passes_fails_and_errors(tmp_path, write_record, capsys):
    write_record(make_record(["x"], name="r1"))
    write_record(make_record(["x"], name="r2"))
    assert main(["check", str(tmp_path)]) == EXIT_OK
    assert capsys.readouterr().out.startswith("OK:")
    write_record(make_record(["q"], name="r3", backend="cuda"))
    assert main(["check", str(tmp_path)]) == EXIT_DIVERGENT
    out = capsys.readouterr().out
    assert out.startswith("FAIL:")
    assert "2 distinct streams" in out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["check", str(empty)]) == EXIT_ERROR
    assert "no .json records" in capsys.readouterr().err


def test_ingest_generic_writes_record_named_after_output(tmp_path, capsys):
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"tokens": ["a", "b"]}), encoding="utf-8")
    out_path = tmp_path / "cpu-run.json"
    code = main([
        "ingest", str(capture), "--format", "generic",
        "--backend", "cpu", "--seed", "42", "--prompt", "p",
        "-o", str(out_path),
    ])
    assert code == EXIT_OK
    record = seedproof.load(str(out_path))
    assert record.name == "cpu-run"
    assert record.config.backend == "cpu"
    assert record.config.seed == 42
    assert record.texts() == ["a", "b"]


def test_ingest_stdout_stdin_and_error_paths(tmp_path, monkeypatch, capsys):
    # Without -o the canonical record goes to stdout.
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"tokens": ["a"]}), encoding="utf-8")
    assert main(["ingest", str(capture), "--format", "generic"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["seedproof_record"] == 1
    # '-' reads the capture from stdin.
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"tokens": ["z"]})))
    assert main(["ingest", "-", "--format", "generic"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["tokens"] == [{"text": "z"}]
    # A malformed capture is a one-line error, exit 2.
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert main(["ingest", str(bad), "--format", "generic"]) == EXIT_ERROR
    assert "invalid JSON" in capsys.readouterr().err
