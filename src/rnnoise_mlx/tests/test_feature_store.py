import json
import hashlib

from rnnoise_mlx.tools.feature_store import _generation_id, publish, verify_generation
from rnnoise_mlx.tools.generate_features import BYTES_PER_SEQUENCE


def source_feature(tmp_path, name="source"):
    source = tmp_path / f"{name}.f32"
    source.write_bytes(b"\0" * BYTES_PER_SEQUENCE)
    manifest = tmp_path / f"{name}.manifest.json"
    manifest.write_text(json.dumps({
        "format_version": 1,
        "kind": "rnnoise-training-features",
        "sequence_count": 1,
        "frames_per_sequence": 2000,
        "values_per_frame": 98,
        "generator": {
            "rng_algorithm": "splitmix64-domain-v1",
            "seed": 7,
            "sequence_start": 0,
            "speech_offset_start": 0,
            "disable_foreground": False,
        },
        "output": {
            "bytes": BYTES_PER_SEQUENCE,
            "sha256": hashlib.sha256(b"\0" * BYTES_PER_SEQUENCE).hexdigest(),
        },
    }))
    return source, manifest


def test_publish_verify_and_reuse(tmp_path):
    source, source_manifest = source_feature(tmp_path)
    destination = tmp_path / "v1" / "train" / "generation-000"
    first = publish(source, source_manifest, destination, 1)
    second = publish(source, source_manifest, destination, 1)
    assert first == second == destination
    assert verify_generation(destination)["sequence_count"] == 1
    index = json.loads((tmp_path / "v1" / "index.json").read_text())
    assert index["generations"][0]["id"] == _generation_id(
        hashlib.sha256(b"\0" * BYTES_PER_SEQUENCE).hexdigest()
    )


def test_verify_rejects_corruption(tmp_path):
    source, source_manifest = source_feature(tmp_path)
    destination = tmp_path / "v1" / "train" / "generation-000"
    publish(source, source_manifest, destination, 1)
    with (destination / "features.f32").open("r+b") as output:
        output.write(b"x")
    try:
        verify_generation(destination)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("corrupt generation was accepted")


def test_reuse_rejects_different_generation_conditions(tmp_path):
    source, source_manifest = source_feature(tmp_path)
    destination = tmp_path / "v1" / "train" / "generation-000"
    publish(source, source_manifest, destination, 1)
    manifest = json.loads(source_manifest.read_text())
    manifest["generator"]["seed"] = 8
    source_manifest.write_text(json.dumps(manifest))
    try:
        publish(source, source_manifest, destination, 1)
    except FileExistsError as error:
        assert "conditions differ" in str(error)
    else:
        raise AssertionError("different generation conditions were reused")


def test_reuse_repairs_missing_index(tmp_path):
    source, source_manifest = source_feature(tmp_path)
    destination = tmp_path / "v1" / "train" / "generation-000"
    publish(source, source_manifest, destination, 1)
    index = tmp_path / "v1" / "index.json"
    index.unlink()

    publish(source, source_manifest, destination, 1)

    repaired = json.loads(index.read_text())
    assert repaired["generations"][0]["id"].startswith("sha256:")


def test_verify_rejects_manifest_schema_mismatch(tmp_path):
    source, source_manifest = source_feature(tmp_path)
    destination = tmp_path / "v1" / "train" / "generation-000"
    publish(source, source_manifest, destination, 1)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["values_per_frame"] = 97
    manifest_path.write_text(json.dumps(manifest))
    try:
        verify_generation(destination)
    except ValueError as error:
        assert "values_per_frame" in str(error)
    else:
        raise AssertionError("unsupported feature manifest was accepted")


def test_verify_rejects_generation_id_not_tied_to_content(tmp_path):
    source, source_manifest = source_feature(tmp_path)
    destination = tmp_path / "v1" / "train" / "generation-000"
    publish(source, source_manifest, destination, 1)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation_id"] = "generation-000"
    manifest_path.write_text(json.dumps(manifest))
    try:
        verify_generation(destination)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("generation ID unrelated to content was accepted")
