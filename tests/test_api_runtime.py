"""Configuration, model identity, and execution routing. No weights needed."""

import json

import pytest

from semif_api import runtime, slots


def config(**overrides):
    base = dict(model="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", backend="torch")
    base.update(overrides)
    return runtime.Config(**base)


def engine(**overrides):
    instance = runtime.Runtime(config(**overrides))
    instance.ready = True
    return instance


def stub(calls, failing=()):
    def run(mode):
        def runner(rows):
            calls.append(mode)
            if mode in failing:
                raise ValueError(f"{mode} refused these rows")
            return [{"id": row["id"]} for row in rows], {"mode": mode}
        return runner
    return run


def rows(count):
    return [{"id": f"q{index}", "state": "s", "question": "?",
             "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]}
            for index in range(count)]


def wire(instance, calls, failing=()):
    make = stub(calls, failing)
    instance._run_direct = make("direct")
    instance._run_serial = make("serial")
    instance._run_shared = make("shared")


def test_model_id_reports_the_weights_that_answer():
    assert runtime.model_id(config()) == "semif/Qwen3.5-4B@851bf6e806ef+torch-bfloat16"
    assert runtime.model_id(config(backend="mlx")) == "semif/Qwen3.5-4B@851bf6e806ef+mlx-bf16"
    assert runtime.model_id(config(backend="mlx", mlx_bits=4)) == "semif/Qwen3.5-4B@851bf6e806ef+mlx-q4"
    assert runtime.model_id(config(dtype="float32")).endswith("+torch-float32")


def test_quantization_is_visible_in_the_reported_identity():
    assert runtime.model_id(config(backend="mlx", mlx_bits=4)) != runtime.model_id(config(backend="mlx"))


def test_jev_aliases_are_accepted_only_when_enabled():
    served = "semif/x@abc+torch-bfloat16"
    assert "jev-latest" in runtime.accepted_names(config(), served)
    assert "jev-latest" not in runtime.accepted_names(config(accept_jev_aliases=False), served)
    assert served in runtime.accepted_names(config(accept_jev_aliases=False), served)


def test_a_single_question_goes_straight_to_direct():
    calls = []
    instance = engine()
    wire(instance, calls)
    _, mode, _, fallback = instance.score(rows(1))
    assert (calls, mode, fallback) == (["direct"], "direct", None)


def test_several_questions_share_one_state_prefill():
    calls = []
    instance = engine()
    wire(instance, calls)
    _, mode, _, fallback = instance.score(rows(3))
    assert (calls, mode, fallback) == (["shared"], "shared", None)


def test_a_refused_prefix_falls_back_and_says_so():
    calls = []
    instance = engine()
    wire(instance, calls, failing={"shared"})
    _, mode, _, fallback = instance.score(rows(3))
    assert calls == ["shared", "serial"]
    assert (mode, fallback) == ("serial", "shared")


def test_direct_is_the_last_resort():
    calls = []
    instance = engine()
    wire(instance, calls, failing={"shared", "serial"})
    _, mode, _, fallback = instance.score(rows(3))
    assert calls == ["shared", "serial", "direct"]
    assert (mode, fallback) == ("direct", "shared")


def test_a_pinned_mode_never_falls_back():
    calls = []
    instance = engine(mode="shared")
    wire(instance, calls, failing={"shared"})
    with pytest.raises(ValueError, match="shared refused"):
        instance.score(rows(3))
    assert calls == ["shared"]


def test_a_pinned_mode_is_used_even_for_one_question():
    calls = []
    instance = engine(mode="serial")
    wire(instance, calls)
    assert instance.score(rows(1))[1] == "serial"


@pytest.mark.parametrize("overrides,message", [
    ({"model": None}, "required"),
    ({"backend": "tensorflow"}, "Backend must be"),
    ({"mode": "guess"}, "Mode must be"),
    ({"max_input_tokens": 0}, "positive"),
    ({"mlx_bits": 4}, "mlx backend"),
    ({"gguf": "/tmp/x.gguf"}, "llamacpp backend"),
    ({"backend": "llamacpp"}, "SEMIF_GGUF"),
    ({"host": "0.0.0.0"}, "SEMIF_API_KEY"),
    ({"prompt_version": slots.V2}, "not implemented"),
])
def test_configuration_is_rejected_before_any_model_loads(overrides, message):
    with pytest.raises(ValueError, match=message):
        config(**overrides).validate()


def test_a_non_loopback_bind_is_allowed_once_a_key_is_set():
    config(host="0.0.0.0", api_key="secret").validate()


def test_temperature_defaults_to_one_so_nothing_is_silently_calibrated():
    assert config().temperature == 1.0
    assert runtime.Config.from_env().temperature == 1.0


def test_a_per_type_temperature_map_is_loaded_from_a_file(tmp_path):
    path = tmp_path / "temperatures.json"
    path.write_text(json.dumps({"default": 1.0, "choice": 1.23, "score": 1.71}))
    instance = config()
    instance.set_temperature(str(path))
    assert instance.temperature_for("choice") == 1.23
    assert instance.temperature_for("score") == 1.71
    assert instance.temperature_for("noul") == 1.0


@pytest.mark.parametrize("payload,message", [
    ({"choice": 0}, "positive"),
    ({"unknown": 1.0}, "unknown keys"),
    ({}, "nonempty"),
])
def test_bad_temperature_files_are_rejected(tmp_path, payload, message):
    path = tmp_path / "temperatures.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        config().set_temperature(str(path))


def test_a_scalar_temperature_clears_any_per_type_map():
    instance = config()
    instance.temperatures = {"choice": 2.0}
    instance.set_temperature("1.5")
    assert instance.temperature_for("choice") == 1.5
