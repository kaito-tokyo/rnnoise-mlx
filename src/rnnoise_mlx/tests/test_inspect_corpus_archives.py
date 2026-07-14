import io
import wave
import zipfile
from pathlib import Path

from rnnoise_mlx.tools.inspect_corpus_archives import flac_info, inspect_archive, speaker_id


def wav_bytes(seconds: int, rate: int = 8000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * rate * seconds)
    return output.getvalue()


def test_speaker_id_uses_middle_filename_field():
    assert speaker_id("irm_02484_00025039317.wav") == "irm:02484"
    assert speaker_id("unknown.wav") is None
    assert speaker_id("train/3000/100/file.flac", "zeroth_korean") == "100"
    assert speaker_id("train/wav/SSB0001/file.wav", "aishell3") == "SSB0001"


def test_flac_info_reads_streaminfo():
    sample_rate = 16000
    channels = 1
    bits = 16
    total_samples = 32000
    packed = (sample_rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | total_samples
    streaminfo = b"\0" * 10 + packed.to_bytes(8, "big") + b"\0" * 16
    payload = b"fLaC" + bytes([0]) + (34).to_bytes(3, "big") + streaminfo
    assert flac_info(io.BytesIO(payload)) == (16000, 1, 16, 32000)


def test_inspect_zip_aggregates_wav_headers(tmp_path: Path):
    path = tmp_path / "speech.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("abc_001_x.wav", wav_bytes(1))
        archive.writestr("abc_002_y.wav", wav_bytes(2))
        archive.writestr("README", b"ignored")
    result = inspect_archive(path)
    assert result["clip_count"] == 2
    assert result["duration_seconds"] == 3
    assert result["speaker_count"] == 2
    assert result["formats"] == [
        {"sample_rate_hz": 8000, "channels": 1, "sample_width_bytes": 2, "compression": "NONE", "clip_count": 2}
    ]
    assert result["header_failures"] == []
