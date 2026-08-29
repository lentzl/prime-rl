import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "publish_q35_2b_hf_latest_remote_v1.py"
SPEC = importlib.util.spec_from_file_location("publish_q35_2b_hf_latest_remote_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeApi:
    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        assert repo_id == "owner/model"
        assert repo_type == "model"
        return [
            ".gitattributes",
            "README.md",
            "config.json",
            "obsolete-index.json",
            "old/model-00001-of-00002.safetensors",
        ]


def test_stale_repo_files_preserves_only_current_checkpoint_and_attributes(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("card")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    stale = MODULE._stale_repo_files(
        _FakeApi(), repo_id="owner/model", checkpoint_dir=tmp_path
    )

    assert stale == [
        "obsolete-index.json",
        "old/model-00001-of-00002.safetensors",
    ]
