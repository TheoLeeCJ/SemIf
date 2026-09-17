"""CLI routing tests with mocked backends: no model downloads, GPU, or MLX needed."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, create_autospec

import pytest

import openjev_phase1
from openjev_phase1 import cli


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    rows = [
        {"id": key, "state": "shared evidence", "question": "Which answer follows?",
         "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}
        for key in ("first", "second")
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "nested" / "output.jsonl"

    def run(mode="direct", *flags):
        monkeypatch.setattr(sys, "argv", [
            "openjev-score", "--mode", mode, "--model", "test/model",
            "--revision", "a" * 40, "--input", str(source), "--output", str(output),
            "--max-tokens", "128", *flags,
        ])
        cli.main()

    return SimpleNamespace(run=run, rows=rows, output=output)


@pytest.fixture
def backends(monkeypatch):
    loaded = (object(), object(), {"revision": "a" * 40})
    results = [{"id": "first", "probabilities": [0.25, 0.75]},
               {"id": "second", "probabilities": [0.8, 0.2]}]
    timing = {"total_seconds": 0.125}

    def mlx_loader(source, revision):
        """Real two-argument signature: stale dtype arguments fail here."""

    mlx = ModuleType("openjev_phase1.mlx_backend")
    mlx.load_causal_model = create_autospec(mlx_loader, return_value=loaded)
    mlx.score = Mock(side_effect=results)
    mlx.score_shared = Mock(return_value=(results, timing))
    mlx.SerialPrefixScorer = Mock(return_value=SimpleNamespace(score=Mock(side_effect=results)))
    monkeypatch.setitem(sys.modules, mlx.__name__, mlx)
    monkeypatch.setattr(openjev_phase1, "mlx_backend", mlx, raising=False)

    torch = SimpleNamespace(
        load_causal_model=Mock(return_value=loaded),
        direct_score=Mock(side_effect=results),
        score_shared=Mock(return_value=(results, timing)),
        SerialPrefixScorer=Mock(return_value=SimpleNamespace(score=Mock(side_effect=results))),
        reranker_score=Mock(side_effect=results),
    )
    for name, mock in vars(torch).items():
        monkeypatch.setattr(cli, name, mock)
    return SimpleNamespace(mlx=mlx, torch=torch, loaded=loaded, results=results, timing=timing)


@pytest.mark.parametrize("mode", ["direct", "serial", "shared"])
def test_mlx_routes_to_matching_scorer(run_cli, backends, mode):
    run_cli.run(mode, "--backend", "mlx")
    mlx = backends.mlx
    model, tokenizer, metadata = backends.loaded
    mlx.load_causal_model.assert_called_once_with("test/model", "a" * 40)
    assert all(mock.call_count == 0 for mock in vars(backends.torch).values())
    if mode == "direct":
        assert mlx.score.call_args_list == [call(model, tokenizer, row, metadata, 128) for row in run_cli.rows]
    elif mode == "serial":
        assert mlx.SerialPrefixScorer.return_value.score.call_args_list == [call(row) for row in run_cli.rows]
    else:
        mlx.score_shared.assert_called_once_with(model, tokenizer, run_cli.rows, metadata, 128)
    expected = backends.results if mode != "shared" else [
        {**result, "shared_timing": backends.timing} for result in backends.results
    ]
    assert [json.loads(line) for line in run_cli.output.read_text().splitlines()] == expected


def test_mlx_rejects_invalid_flags_before_loading(run_cli, backends, capsys):
    with pytest.raises(SystemExit) as error:
        run_cli.run("reranker", "--backend", "mlx")
    assert error.value.code == 2
    assert "MLX requires" in capsys.readouterr().err
    backends.mlx.load_causal_model.assert_not_called()
    backends.torch.load_causal_model.assert_not_called()
    assert not run_cli.output.exists()


def test_torch_reranker_is_cuda_only(run_cli, backends, capsys):
    run_cli.run("reranker")
    backends.torch.load_causal_model.assert_called_once_with("test/model", "a" * 40, "cuda", "bfloat16")
    model, tokenizer, metadata = backends.loaded
    assert backends.torch.reranker_score.call_args_list == [
        call(model, tokenizer, row, metadata, 128) for row in run_cli.rows
    ]
    run_cli.output.unlink()
    with pytest.raises(SystemExit) as error:
        run_cli.run("reranker", "--device", "mps")
    assert error.value.code == 2
    assert "Reranker mode requires CUDA" in capsys.readouterr().err
    assert backends.torch.reranker_score.call_count == 2


def test_existing_output_is_not_overwritten(run_cli, backends, capsys):
    original = b"existing benchmark evidence\n"
    run_cli.output.parent.mkdir(parents=True)
    run_cli.output.write_bytes(original)
    with pytest.raises(SystemExit) as error:
        run_cli.run("direct")
    assert error.value.code == 2
    assert "Output must be new" in capsys.readouterr().err
    assert run_cli.output.read_bytes() == original
    backends.torch.load_causal_model.assert_not_called()
    backends.mlx.load_causal_model.assert_not_called()
