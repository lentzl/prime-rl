import importlib.util
import json
from argparse import Namespace
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


def test_remote_publish_streams_credential_outside_command(monkeypatch) -> None:
    calls = []
    args = Namespace(
        remote_upload_helper="/repo/scripts/publish.py",
        remote_uv_bin="/home/ubuntu/.local/bin/uv",
    )

    def fake_ssh(_args, command: str, *, input_text: str | None = None) -> str:
        calls.append((command, input_text))
        return ""

    monkeypatch.setattr(MODULE, "_ssh", fake_ssh)
    MODULE._upload_remote_checkpoint(
        args,
        repo_id="owner/latest",
        model_path="/outputs/cycle 61/weights/step_1",
        commit_message="replace latest",
        token="hf_secret",
        model_card="# latest",
    )

    command, input_text = calls[0]
    assert "hf_secret" not in command
    assert "'/outputs/cycle 61/weights/step_1'" in command
    assert json.loads(input_text) == {
        "model_card": "# latest",
        "token": "hf_secret",
    }


def test_prune_never_deletes_newer_update_with_admission_pending(monkeypatch) -> None:
    old = "/outputs/grpo-auto-000054-child/weights/step_1"
    protected_child = "/outputs/grpo-auto-000056-child/weights/step_1"
    pending_child = "/outputs/grpo-auto-000058-child/weights/step_1"
    protected_coordinator = "/outputs/grpo-auto-000057-coordinator/weights/step_1"
    events = [
        {
            "kind": "train_completed",
            "cycle": 54,
            "role": "child",
            "output_candidate": {"model_path": old, "model_sha256": "old"},
        },
        {"kind": "evaluation_completed", "cycle": 54, "role": "child"},
        {
            "kind": "train_completed",
            "cycle": 56,
            "role": "child",
            "output_candidate": {
                "model_path": protected_child,
                "model_sha256": "child",
            },
        },
        {"kind": "evaluation_completed", "cycle": 56, "role": "child"},
        {
            "kind": "train_completed",
            "cycle": 58,
            "role": "child",
            "output_candidate": {
                "model_path": pending_child,
                "model_sha256": "pending",
            },
        },
    ]
    frontiers = {
        "child": (
            {},
            {
                "cycle": 56,
                "output_candidate": {"model_path": protected_child},
            },
            {},
        ),
        "coordinator": (
            {},
            {
                "cycle": 57,
                "output_candidate": {"model_path": protected_coordinator},
            },
            {},
        ),
    }
    commands = []

    def fake_ssh(_args, command: str) -> str:
        commands.append(command)
        return old + "\n"

    monkeypatch.setattr(MODULE, "_ssh", fake_ssh)
    removed = MODULE._prune_superseded_remote_weights(Namespace(), events, frontiers)

    assert [item["model_path"] for item in removed] == [old]
    assert pending_child not in commands[0]
    assert protected_child not in commands[0]


def test_prune_never_deletes_failed_admission_frontier(monkeypatch) -> None:
    protected_child = "/outputs/grpo-auto-000056-child/weights/step_1"
    failed_child = "/outputs/grpo-auto-000058-child/weights/step_1"
    protected_coordinator = "/outputs/grpo-auto-000057-coordinator/weights/step_1"
    events = [
        {
            "kind": "train_completed",
            "cycle": 58,
            "role": "child",
            "output_candidate": {
                "model_path": failed_child,
                "model_sha256": "failed",
            },
        },
        {"kind": "evaluation_failed", "cycle": 58, "role": "child"},
    ]
    frontiers = {
        "child": (
            {},
            {
                "cycle": 56,
                "output_candidate": {"model_path": protected_child},
            },
            {},
        ),
        "coordinator": (
            {},
            {
                "cycle": 57,
                "output_candidate": {"model_path": protected_coordinator},
            },
            {},
        ),
    }

    def unexpected_ssh(_args, _command: str) -> str:
        raise AssertionError("failed-admission frontier must not be pruned")

    monkeypatch.setattr(MODULE, "_ssh", unexpected_ssh)
    assert MODULE._prune_superseded_remote_weights(Namespace(), events, frontiers) == []
