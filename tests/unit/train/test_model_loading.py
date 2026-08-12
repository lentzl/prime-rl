from types import SimpleNamespace

from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.trainer import model as model_module


def test_pre_download_model_forwards_revision(monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/model-snapshot"

    monkeypatch.setattr(model_module, "snapshot_download", snapshot_download)

    model_module.pre_download_model("PrimeIntellect/test-model", revision="pinned-revision")

    assert calls == [
        {
            "repo_id": "PrimeIntellect/test-model",
            "repo_type": "model",
            "revision": "pinned-revision",
        }
    ]


def test_setup_tokenizer_forwards_revision(monkeypatch):
    tokenizer = SimpleNamespace(eos_token_id=42, pad_token_id=None)
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return tokenizer

    monkeypatch.setattr(model_module.AutoTokenizer, "from_pretrained", from_pretrained)

    result = model_module.setup_tokenizer(
        TokenizerConfig(
            name="PrimeIntellect/test-model",
            revision="pinned-revision",
            trust_remote_code=True,
        )
    )

    assert result is tokenizer
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert calls == [
        (
            "PrimeIntellect/test-model",
            {
                "revision": "pinned-revision",
                "trust_remote_code": True,
            },
        )
    ]
