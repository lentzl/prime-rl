import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/sync_q35_2b_latest_hf_v1.py"
SPEC = importlib.util.spec_from_file_location("sync_q35_latest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def delete_repo(self, repo_id: str, *, repo_type: str) -> None:
        self.calls.append(("delete", (repo_id, repo_type)))

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool) -> None:
        self.calls.append(("create", (repo_id, repo_type, private)))

    def upload_folder(self, **kwargs: object) -> None:
        self.calls.append(("upload", kwargs))


def test_latest_only_publish_recreates_slot_before_upload(tmp_path: Path) -> None:
    api = _FakeApi()

    MODULE._upload_latest_only(api, "owner/latest", tmp_path, "replace")

    assert [name for name, _ in api.calls] == ["delete", "create", "upload"]
    assert api.calls[1][1] == ("owner/latest", "model", True)
    upload = api.calls[2][1]
    assert upload["repo_id"] == "owner/latest"
    assert upload["folder_path"] == tmp_path
    assert upload["commit_message"] == "replace"
