# SpeexDSP Common Voice experiment

This is an independent training experiment. It replaces the learned CJK
cleaner with the fixed SpeexDSP preprocessor while retaining the RNNoise
feature extraction, noise mixture, RIR, and MLX training code. No checkpoint,
feature file, prepared PCM, or manifest is copied from an earlier experiment.
All inputs are read from the Dataset root and all outputs go under a new
experiment directory.

## Mac mini preparation

Run from `/Users/umireon/work/prodesk-wsl/rnnoise-mlx` with `/opt/homebrew/bin`
on `PATH`. Set these paths before starting; do not point them at an old
Worktree:

```sh
DATASET=/Users/umireon/Datasets/rnnoise-mlx-multilingual-corpus-20260713
EXP=/Users/umireon/work/prodesk-wsl/rnnoise-mlx-experiments/speexdsp-ja-minus12-final-90h-10k
mkdir -p "$EXP"/{source,manifests,prepared,offsets,features,logs}
```

Keep the original Common Voice archive under `$DATASET` read-only. The
Dataset copy is the only source of clips and corpus metadata.

Select speaker-disjoint Japanese clips, extract them into a separate directory,
measure speech onsets, and clean only accepted clips:

```sh
python3 -m rnnoise_mlx.tools.select_common_voice_clips \
  "$DATASET/api/common-voice-scripted-26.0/ja/<archive>.tar.gz" "$EXP/manifests/cv-ja-selection.json" \
  --train-hours 8 --eval-minutes 25 --speaker-cap-minutes 5 --seed 141
python3 -m rnnoise_mlx.tools.extract_common_voice_selection \
  "$DATASET/api/common-voice-scripted-26.0/ja/<archive>.tar.gz" "$EXP/manifests/cv-ja-selection.json" \
  "$EXP/source/cv-ja-selected"
python3 -m rnnoise_mlx.tools.analyze_speech_onsets \
  "$EXP/source/cv-ja-selected" "$EXP/manifests/cv-ja-onsets" \
  --selection-manifest "$EXP/source/cv-ja-selected/extraction-manifest.json" \
  --filter-threshold -40 --minimum-onset-ms 250 --workers 8
python3 -m rnnoise_mlx.tools.cleanup_common_voice_ja \
  "$EXP/source/cv-ja-selected" "$EXP/prepared/common-voice-ja-speex12" \
  --filter-manifest "$EXP/manifests/cv-ja-onsets/filter-manifest.json" \
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
  configs/speexdsp-final-90h.example.json "$EXP/prepared/mix" \
  --augmentation-prepared "$EXP/prepared/augmentation"
python3 -m rnnoise_mlx.tools.select_speech_offsets \
  "$EXP/prepared/mix/train_speech.pcm" "$EXP/offsets/train.txt" \
  --count 20000 --seed 141
python3 -m rnnoise_mlx.tools.select_speech_offsets \
  "$EXP/prepared/mix/eval_speech.pcm" "$EXP/offsets/eval.txt" \
  --count 500 --seed 142
python3 -m rnnoise_mlx.tools.build_dump_features
python3 -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features "$EXP/prepared/mix" "$EXP/features" \
  --train-count 20000 --eval-count 500 --speech-offsets "$EXP/offsets" \
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
