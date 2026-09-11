# SpeexDSP Common Voice experiment

This experiment replaces the learned CJK cleaner with the fixed SpeexDSP
preprocessor. The existing RNNoise feature extraction, noise mixture, RIR,
and MLX training pipeline remain unchanged.

## Mac mini preparation

Run from `/Users/umireon/work/prodesk-wsl/rnnoise-mlx` with `/opt/homebrew/bin`
on `PATH`. Keep the original Common Voice archive read-only. The archive and
dataset manifest are in the shared Drive folder `common-voice-scripted-26.0`.

Select speaker-disjoint Japanese clips, extract them into a separate directory,
measure speech onsets, and clean only accepted clips:

```sh
python3 -m rnnoise_mlx.tools.select_common_voice_clips \
  data/source/common-voice-ja.tar.gz data/manifests/cv-ja-selection.json \
  --train-hours 8 --eval-minutes 25 --speaker-cap-minutes 5 --seed 141
python3 -m rnnoise_mlx.tools.extract_common_voice_selection \
  data/source/common-voice-ja.tar.gz data/manifests/cv-ja-selection.json \
  data/cv-ja-selected
python3 -m rnnoise_mlx.tools.analyze_speech_onsets \
  data/cv-ja-selected data/manifests/cv-ja-onsets \
  --selection-manifest data/cv-ja-selected/extraction-manifest.json \
  --filter-threshold -40 --minimum-onset-ms 250 --workers 8
python3 -m rnnoise_mlx.tools.cleanup_common_voice_ja \
  data/cv-ja-selected data/common-voice-ja-speex12 \
  --filter-manifest data/manifests/cv-ja-onsets/filter-manifest.json \
  --speex-library /opt/homebrew/lib/libspeexdsp.dylib \
  --noise-suppress-db -12 --threshold -40 --margin-ms 150 \
  --workers 4 --resume
```

The extraction manifest produced by `extract_common_voice_selection` should
be used as the selection input for onset analysis. Preserve every manifest and
the archive SHA-256. The exact output directory names must match the SpeexDSP
90-hour mix specification.

Prepare the existing augmentation population and render the weighted speech
mix with `configs/speexdsp-final-90h.example.json`, then build the vendored
feature extractor and generate 20,000 train plus 500 evaluation sequences:

```sh
python3 -m rnnoise_mlx.tools.prepare_speech_mix \
  configs/speexdsp-final-90h.example.json data/prepared \
  --augmentation-prepared /path/to/augmentation-prepared
python3 -m rnnoise_mlx.tools.select_speech_offsets \
  data/prepared/train_speech.pcm data/offsets/train.txt \
  --count 20000 --seed 141
python3 -m rnnoise_mlx.tools.select_speech_offsets \
  data/prepared/eval_speech.pcm data/offsets/eval.txt \
  --count 500 --seed 142
python3 -m rnnoise_mlx.tools.build_dump_features
python3 -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features data/prepared data/features \
  --train-count 20000 --eval-count 500 --speech-offsets data/offsets \
  --seed 141
```

Run the expensive Mac mini stages through `task-spooler` in the order shown;
do not start feature generation before the prepared speech manifest is
complete. Verify both feature manifests and publish the feature files to the
shared store only after their sizes and SHA-256 values match.

## Colab training

Copy `train.f32`, `eval.f32`, and their manifests to `/content` before running
MLX. Start the existing Colab helper from WSL so the reverse MLflow tunnel is
active:

```sh
./colab/start_session.sh --auth adc --session speexdsp-ja-minus12 --gpu L4
```

Run training in the connected Colab shell:

```sh
python3 -m rnnoise_mlx.training.train /content/train.f32 \
  /content/runs/speexdsp-ja-minus12-final-90h-10k \
  --eval-features /content/eval.f32 \
  --batch-size 8 --sequence-length 2000 \
  --segmented-tbptt-length 500 --segmented-tbptt-state carry \
  --max-updates 10000 --checkpoint-every 500 --seed 141 \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-experiment rnnoise-speexdsp-common-voice \
  --mlflow-run-name speexdsp-ja-minus12-final-90h-10k \
  --provenance-artifact /content/train.manifest.json \
  --provenance-artifact /content/eval.manifest.json
```

Upload or retain each complete checkpoint through MLflow. If the runtime ends,
download the latest committed checkpoint, copy it to `/content`, and resume
with the same MLflow run ID. Keep Alive extensions are not part of this
workflow.

## Acceptance

The experiment is successful when the full preprocessing and feature pipeline
completes, 10,000 updates finish or resume to completion, checkpoints exist at
500-update boundaries, and the MLflow run contains the data, SpeexDSP, feature,
and training provenance. This is a construction-feasibility experiment, not a
parameter benchmark.
