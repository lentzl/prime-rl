from types import SimpleNamespace

import pytest

from prime_rl.trainer.ckpt import (
    CHAT_EOS_TOKEN,
    _chat_eos_token_id,
    _numeric_eos_fields,
    _save_processor_metadata,
    _set_numeric_eos_token_ids,
    _validate_chat_eos_metadata,
)


class ChatTokenizer:
    chat_template = "{{ message }}<|im_end|>"
    eos_token_id = 248046

    def encode(self, text, *, add_special_tokens):
        assert text == CHAT_EOS_TOKEN
        assert add_special_tokens is False
        return [248046]


class SavingComponent:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def save_pretrained(self, path) -> None:
        (path / self.filename).write_text("{}")


class SavingProcessor(SavingComponent):
    def __init__(self) -> None:
        super().__init__("processor_config.json")
        self.image_processor = SavingComponent("preprocessor_config.json")
        self.video_processor = SavingComponent("video_preprocessor_config.json")


def test_chat_eos_token_id_requires_template_usage() -> None:
    assert _chat_eos_token_id(ChatTokenizer()) == 248046
    assert _chat_eos_token_id(SimpleNamespace(chat_template="{{ message }}")) is None


def test_set_numeric_eos_token_ids_updates_nested_configs() -> None:
    config = SimpleNamespace(
        eos_token_id=248044,
        text_config=SimpleNamespace(eos_token_id=248044),
        unrelated={"eos_token_id": "248044"},
    )

    _set_numeric_eos_token_ids(config, 248046)

    assert config.eos_token_id == 248046
    assert config.text_config.eos_token_id == 248046
    assert config.unrelated["eos_token_id"] == "248044"


def test_validate_chat_eos_metadata_rejects_any_nested_mismatch(tmp_path) -> None:
    (tmp_path / "config.json").write_text('{"text_config": {"eos_token_id": 248044}}')
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 248046}')

    with pytest.raises(ValueError, match="config.json:text_config.eos_token_id=248044"):
        _validate_chat_eos_metadata(tmp_path, ChatTokenizer(), 248046)


def test_validate_chat_eos_metadata_rejects_malformed_json(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{")

    with pytest.raises(ValueError):
        _validate_chat_eos_metadata(tmp_path, ChatTokenizer(), 248046)


def test_numeric_eos_fields_ignores_strings_and_bools() -> None:
    assert _numeric_eos_fields(
        {"eos_token_id": 248046, "nested": {"eos_token_id": "248046"}, "flag": {"eos_token_id": True}}
    ) == [("eos_token_id", 248046)]


def test_save_processor_metadata_includes_standalone_multimodal_components(tmp_path) -> None:
    _save_processor_metadata(SavingProcessor(), tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    }
