import sys

import pytest

from semif_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--backend", "mlx", "--mode", "reranker"], "reranker requires torch"),
    (["--mode", "direct", "--mlx-bits", "4"], "requires --backend mlx"),
    (["--mode", "direct", "--mlx-cache-limit-mib", "0"], "requires --backend mlx"),
    (["--mode", "direct", "--backend", "mlx", "--mlx-cache-limit-mib", "-1"], "must be nonnegative"),
    (["--backend", "ryzenai-npu", "--mode", "serial"], "direct mode only"),
    (["--backend", "ryzenai-npu", "--mode", "shared"], "direct mode only"),
    (["--backend", "ryzenai-npu", "--mode", "reranker"], "direct mode only"),
])
def test_invalid_backend_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["semif-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize('limit', [None, 0, 512])
def test_cli_passes_cache_limit_to_loader(tmp_path, monkeypatch, limit):
    import json
    from types import SimpleNamespace
    import semif_phase1

    fake_backend = SimpleNamespace(
        DEFAULT_CACHE_LIMIT_MIB=256,
        load_model=lambda source, revision, bits, *, cache_limit_mib:
            (None, None, {'limit': cache_limit_mib}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(semif_phase1, 'mlx_backend', fake_backend, raising=False)
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    source.write_text(json.dumps({'id': 'test', 'state': 'Evidence', 'question': 'Supported?',
                                 'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}) + '\n')
    args = ['semif-score', '--backend', 'mlx', '--mode', 'direct', '--model', 'unused',
            '--revision', 'unused', '--input', str(source), '--output', str(output)]
    if limit is not None:
        args += ['--mlx-cache-limit-mib', str(limit)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert json.loads(output.read_text())['limit'] == (256 if limit is None else limit)


def test_cli_npu_dispatch_preserves_utf8_and_never_loads_torch(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import semif_phase1
    import semif_phase1.cli as cli

    calls = []

    def load(source, revision):
        calls.append((source, revision))
        return "npu-model", "tokenizer", {"backend": "ryzenai-npu"}

    def score(model, tokenizer, row, metadata, max_tokens):
        assert (model, tokenizer, max_tokens) == ("npu-model", "tokenizer", 512)
        assert row["state"] == "更新は成功しました。"
        return {"id": row["id"], "model": metadata, "probabilities": [0.9, 0.1]}

    def unexpected_gpu_load(*args):
        pytest.fail("NPU CLI must never enter the torch model loader")

    monkeypatch.setattr(cli, "load_causal_model", unexpected_gpu_load)
    monkeypatch.setattr(semif_phase1, "ryzenai_backend", SimpleNamespace(load_model=load, score=score), raising=False)
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({
        "id": "日本語", "state": "更新は成功しました。", "question": "成功しましたか？",
        "options": [{"id": "yes", "description": "成功"}, {"id": "no", "description": "失敗"}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "ryzenai-npu", "--mode", "direct", "--model", "local-model",
        "--revision", "pinned-revision", "--input", str(source), "--output", str(output), "--max-tokens", "512",
    ])
    main()
    assert calls == [("local-model", "pinned-revision")]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "id": "日本語", "model": {"backend": "ryzenai-npu"}, "probabilities": [0.9, 0.1],
    }
    with pytest.raises(SystemExit):
        main()
    assert len(calls) == 1  # Existing output rejected before loading the model again.
