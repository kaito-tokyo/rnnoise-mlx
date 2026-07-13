"""Build the vendored RNNoise feature extractor."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()

    vendor = project_root() / "Vendors/xiph-rnnoise"
    output = args.output or vendor / "dump_features"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        "dump_features.c",
        "denoise.c",
        "pitch.c",
        "celt_lpc.c",
        "kiss_fft.c",
        "parse_lpcnet_weights.c",
        "rnnoise_tables.c",
    ]
    subprocess.run(
        [
            "clang",
            "-O3",
            "-DTRAINING",
            f"-I{vendor}",
            f"-I{vendor / 'src'}",
            f"-I{vendor / 'include'}",
            *(str(vendor / "src" / source) for source in sources),
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
    )
    probe = subprocess.run([str(output)], text=True, capture_output=True)
    if not (probe.stdout + probe.stderr).startswith("usage:"):
        raise SystemExit("built dump_features did not print its usage message")
    print(f"built {output} from Vendors/xiph-rnnoise")


if __name__ == "__main__":
    main()
