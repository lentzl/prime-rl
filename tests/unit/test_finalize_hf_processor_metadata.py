import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "finalize_hf_processor_metadata.py"
SPEC = importlib.util.spec_from_file_location("finalize_hf_processor_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_model_dirs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vision_config": {"model_type": "qwen3_5_vision_encoder"},
    }
    _write_json(source / "config.json", config)
    _write_json(destination / "config.json", config)
    _write_json(source / "preprocessor_config.json", {"size": 28})
    _write_json(source / "video_preprocessor_config.json", {"fps": 2})
    return source, destination


def test_finalize_copies_required_processor_metadata(tmp_path: Path) -> None:
    source, destination = _make_model_dirs(tmp_path)

    files = MODULE.finalize(source, destination)

    assert files == ["preprocessor_config.json", "video_preprocessor_config.json"]
    assert json.loads((destination / "preprocessor_config.json").read_text()) == {
        "size": 28
    }
    assert MODULE.finalize(source, destination) == files


def test_finalize_preserves_checkpoint_owned_metadata(tmp_path: Path) -> None:
    source, destination = _make_model_dirs(tmp_path)
    _write_json(destination / "preprocessor_config.json", {"size": 14})

    with pytest.raises(MODULE.MetadataFailure, match="differs from source"):
        MODULE.finalize(source, destination)


def test_finalize_rejects_incomplete_multimodal_source(tmp_path: Path) -> None:
    source, destination = _make_model_dirs(tmp_path)
    (source / "preprocessor_config.json").unlink()
    (source / "video_preprocessor_config.json").unlink()

    with pytest.raises(MODULE.MetadataFailure, match="no processor metadata"):
        MODULE.finalize(source, destination)


def test_finalize_accepts_consolidated_processor_metadata(tmp_path: Path) -> None:
    source, destination = _make_model_dirs(tmp_path)
    (source / "preprocessor_config.json").unlink()
    (source / "video_preprocessor_config.json").unlink()
    _write_json(source / "processor_config.json", {"processor_class": "Qwen3VLProcessor"})

    files = MODULE.finalize(source, destination)

    assert files == ["processor_config.json"]
