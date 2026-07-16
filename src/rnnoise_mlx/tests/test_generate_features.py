import hashlib

from rnnoise_mlx.tools.generate_features import _input_record, _rir_records


def test_input_records_hash_pcm_and_referenced_rirs(tmp_path):
    pcm = tmp_path / "speech.pcm"
    pcm.write_bytes(b"speech")
    assert _input_record(pcm)["sha256"] == hashlib.sha256(b"speech").hexdigest()

    rir = tmp_path / "room.f32"
    rir.write_bytes(b"rir")
    rir_list = tmp_path / "rir-list.txt"
    rir_list.write_text(f"{rir}\n")
    assert _rir_records(rir_list) == [
        {
            "path": str(rir.resolve()),
            "bytes": 3,
            "sha256": hashlib.sha256(b"rir").hexdigest(),
        }
    ]
