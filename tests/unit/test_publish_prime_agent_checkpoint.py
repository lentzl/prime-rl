import json

import pytest

pytest.importorskip("huggingface_hub")
pytest.importorskip("transformers")

from scripts.publish_prime_agent_checkpoint import (
    eos_fields,
    normalize_eos_fields,
    repair_checkpoint_eos,
    validate_checkpoint,
    validate_remote_file_list,
)


def test_eos_fields_finds_nested_numeric_values() -> None:
    assert eos_fields(
        {
            "eos_token_id": 248046,
            "text_config": {"eos_token_id": 248046},
            "ignored": {"eos_token_id": "248046"},
        }
    ) == [
        ("eos_token_id", 248046),
        ("text_config.eos_token_id", 248046),
    ]


def test_validate_checkpoint_rejects_adapter_only_directory(tmp_path) -> None:
    (tmp_path / "STABLE").touch()
    (tmp_path / "adapter_model.safetensors").touch()

    with pytest.raises(ValueError, match="merged full-model safetensors"):
        validate_checkpoint(tmp_path)


def test_validate_checkpoint_rejects_mismatched_nested_eos(monkeypatch, tmp_path) -> None:
    class Tokenizer:
        eos_token_id = 248046

        @staticmethod
        def encode(text, *, add_special_tokens):
            assert text == "<|im_end|>"
            assert add_special_tokens is False
            return [248046]

    (tmp_path / "STABLE").touch()
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 248044}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 248046}))
    monkeypatch.setattr(
        "scripts.publish_prime_agent_checkpoint.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )

    with pytest.raises(ValueError, match="text_config.eos_token_id=248044"):
        validate_checkpoint(tmp_path)


def test_normalize_eos_fields_updates_nested_scalars_and_lists() -> None:
    metadata = {
        "eos_token_id": 248044,
        "text_config": {"eos_token_id": [248044, 248046]},
        "ignored": {"eos_token_id": "248044"},
    }

    assert normalize_eos_fields(metadata, 248046) == 2
    assert metadata == {
        "eos_token_id": 248046,
        "text_config": {"eos_token_id": [248046, 248046]},
        "ignored": {"eos_token_id": "248044"},
    }


def test_repair_checkpoint_eos_is_atomic_with_stable_marker(monkeypatch, tmp_path) -> None:
    class Tokenizer:
        eos_token_id = 248046

        @staticmethod
        def encode(text, *, add_special_tokens):
            assert text == "<|im_end|>"
            assert add_special_tokens is False
            return [248046]

    (tmp_path / "STABLE").touch()
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 248044}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 248044}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token_id": 248044}))
    monkeypatch.setattr(
        "scripts.publish_prime_agent_checkpoint.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )

    result = repair_checkpoint_eos(tmp_path)

    assert result["repaired_eos_fields"] == 3
    assert (tmp_path / "STABLE").is_file()
    assert json.loads((tmp_path / "config.json").read_text())["text_config"]["eos_token_id"] == 248046
    assert json.loads((tmp_path / "generation_config.json").read_text())["eos_token_id"] == 248046
    assert json.loads((tmp_path / "tokenizer_config.json").read_text())["eos_token_id"] == 248046


def test_remote_file_list_requires_merged_model_and_metadata() -> None:
    valid = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "prime_agent_bundle.json",
        "model-00001-of-00002.safetensors",
    }
    validate_remote_file_list(valid)

    with pytest.raises(RuntimeError, match="full-model safetensors"):
        validate_remote_file_list((valid - {"model-00001-of-00002.safetensors"}) | {"adapter_model.safetensors"})
