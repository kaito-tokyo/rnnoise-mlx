import hashlib
import sys

from rnnoise_mlx.tools.generate_features import _input_record, _rir_records, main


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


def test_main_keeps_evaluation_foreground_distribution_fixed(tmp_path, monkeypatch):
    dump_features = tmp_path / "dump_features"
    dump_features.write_bytes(b"")
    calls = []
    monkeypatch.setattr(
        "rnnoise_mlx.tools.generate_features.generate",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_features",
            str(dump_features),
            str(tmp_path / "prepared"),
            str(tmp_path / "output"),
            "--train-count", "2",
            "--eval-count", "1",
            "--foreground-probability-denominator", "4",
        ],
    )
    main()
    assert calls[0][0][-1] == 4
    assert calls[1][0][-1] == 8
