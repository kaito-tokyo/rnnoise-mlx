# CJK two-stage training pipeline

The first model is a conservative corpus cleaner trained only from clean CJK
speech. It cleans selected Common Voice Japanese clips. The second model is the
distributable denoiser trained from a frozen speech-mixture specification that
points at those cleaned clips. The two models never share feature files.

## Frozen decisions

- Cleaner speech: Kokoro 20%, AISHELL-3 40%, Zeroth Korean 40%.
- Cleaner train population: 15 hours (Kokoro 3 h, AISHELL-3 6 h, Zeroth Korean 6 h).
- Cleaner training: 2,000-frame sequences, carry-state TBPTT 250, 10,000 updates.
- Cleaner augmentation: background noise and RIR enabled; foreground speech disabled.
- Speech offsets: SplitMix64, seed 141 for train and 142 for evaluation.
- Common Voice cleaning is file-preserving and resumable, with input, output,
  and compiled-model hashes recorded.
- AISHELL-3 and Zeroth Korean remain original speech in stage two.

The final stage-two proportions are supplied as a JSON specification. Changing
them creates a different experiment and must not reuse old feature files.

## Persistent experiment checkout

Long jobs run from a purpose-named checkout under `~/Worktrees`, not a
Codex-managed worktree. The commands below assume `ROOT` is that checkout.

```sh
ROOT=/Users/umireon/Worktrees/rnnoise-mlx-cjk-cleaner-20260714
BASE_PREPARED=/Users/umireon/Worktrees/rnnoise-mlx-base-10k-20260712-rerun2/data/prepared
DATASET=/Users/umireon/Datasets/rnnoise-mlx-multilingual-corpus-20260713

cd "$ROOT"
python -m rnnoise_mlx.tools.build_dump_features
swift build -c release --product rnnoise-mlx-denoise
```

Copy `configs/cjk-cleaner.example.json` into the experiment directory and
replace its Kokoro paths with the actual deterministic train/evaluation split.

## Stage 1: CJK cleaner

Build exact CJK PCM and link the unchanged augmentation population:

```sh
python -m rnnoise_mlx.tools.prepare_speech_mix \
  cjk-cleaner-mix.json data/cjk-cleaner/prepared \
  --augmentation-prepared "$BASE_PREPARED"

python -m rnnoise_mlx.tools.select_speech_offsets \
  data/cjk-cleaner/prepared/train_speech.pcm data/cjk-cleaner/offsets/train.txt \
  --count 10000 --seed 141
python -m rnnoise_mlx.tools.select_speech_offsets \
  data/cjk-cleaner/prepared/eval_speech.pcm data/cjk-cleaner/offsets/eval.txt \
  --count 500 --seed 142

python -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features data/cjk-cleaner/prepared \
  data/cjk-cleaner/features --train-count 10000 --eval-count 500 \
  --speech-offsets data/cjk-cleaner/offsets --disable-foreground
```

Train with immutable 1,000-update generations:

```sh
python -m rnnoise_mlx.training.train \
  data/cjk-cleaner/features/train.f32 runs/cjk-cleaner-10k \
  --eval-features data/cjk-cleaner/features/eval.f32 \
  --batch-size 8 --sequence-length 2000 \
  --segmented-tbptt-length 250 --segmented-tbptt-state carry \
  --max-updates 10000 --checkpoint-every 1000 --seed 141 \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment "$MLFLOW_EXPERIMENT" \
  --mlflow-run-name cjk-cleaner-tbptt250-10k
```

The 10k checkpoint is the fixed cleaner candidate. Intermediate checkpoints
remain diagnostic artifacts, but are not alternative production choices. If
the 10k listening gate fails, stop before Common Voice cleanup and revise the
cleaner experiment rather than silently selecting an earlier checkpoint.

```sh
python -m rnnoise_mlx.tools.export_bnns_graph \
  runs/cjk-cleaner-10k/model.safetensors runs/cjk-cleaner-10k/RNNoiseGraph.mlpackage
xcrun coremlcompiler compile runs/cjk-cleaner-10k/RNNoiseGraph.mlpackage \
  runs/cjk-cleaner-10k --platform macOS --deployment-target 15.0
```

Measure leading speech margins on the already extracted eight-hour Common
Voice selection. Keep clips whose -40 dBFS onset is at least 250 ms; do not
refill from the later ten-hour selection.

```sh
python -m rnnoise_mlx.tools.analyze_speech_onsets \
  "$DATASET/prepared/cv26-ja-selection" data/cv26-ja-onsets \
  --selection-manifest "$DATASET/api/common-voice-scripted-26.0/ja/clip-selection.json" \
  --filter-threshold -40 --minimum-onset-ms 250 --workers 8

python -m rnnoise_mlx.tools.clean_audio_corpus \
  "$DATASET/prepared/cv26-ja-selection" data/cv26-ja-clean-full \
  --executable .build/release/rnnoise-mlx-denoise \
  --model runs/cjk-cleaner-10k/RNNoiseGraph.mlmodelc \
  --filter-manifest data/cv26-ja-onsets/filter-manifest.json --workers 4 --resume

python -m rnnoise_mlx.tools.trim_audio_corpus \
  data/cv26-ja-clean-full data/cv26-ja-clean \
  --filter-manifest data/cv26-ja-onsets/filter-manifest.json \
  --threshold -40 --margin-ms 150 --workers 4 --resume
```

The full leading margin is available to the cleaner, but only 150 ms before
the measured onset is retained for stage-two clean speech. The filtered and
trimmed training population is shorter than eight unique hours, so the final
mix specification opts the Japanese training source into deterministic
stable-order repetition. Evaluation does not repeat its Japanese source.

Do not proceed automatically. Compare original and cleaned fixed Japanese
clips, especially unvoiced vowels, `/ɕ/`, geminates, moraic nasals, and word
edges. A cleaner that suppresses more noise but changes these sounds fails.

## Stage 2: distributable model

Copy `configs/final-base-90h.example.json` to `final-base-mix.json`. Its train
population is fixed at 90 hours: 20 hours English, 50 hours official
multilingual speech, 8 hours cleaned Japanese, 6 hours AISHELL-3, and 6 hours
Zeroth Korean. The first 70 hours are the already verified prefix of the old
100-hour PCM; the removed Vietnamese and Arabic blocks occur later and are not
read. Evaluation is language-balanced separately at English 40% and Japanese,
Mandarin, and Korean 20% each.

This stage intentionally preserves the previous production-training method and
the exact 16-language official 50-hour population. Vietnamese and Arabic alone
are removed because maintaining sufficiently clean, commercially usable source
populations is impractical for this run and would weaken the purpose of their
inclusion. Their ten hours are not reassigned; changing other language weights
would add a second corpus-mixture variable to the Common Voice cleanup test.

```sh
python -m rnnoise_mlx.tools.prepare_speech_mix \
  final-base-mix.json data/final-base/prepared \
  --augmentation-prepared "$BASE_PREPARED"
```

Generate new offsets and features. Do not use `--disable-foreground` here
unless a separate final-model experiment intentionally tests that change; the
default preserves upstream's approximately one-in-eight foreground mixing.

```sh
python -m rnnoise_mlx.tools.select_speech_offsets \
  data/final-base/prepared/train_speech.pcm data/final-base/offsets/train.txt \
  --count 20000 --seed 141
python -m rnnoise_mlx.tools.select_speech_offsets \
  data/final-base/prepared/eval_speech.pcm data/final-base/offsets/eval.txt \
  --count 500 --seed 142
python -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features data/final-base/prepared \
  data/final-base/features --train-count 20000 --eval-count 500 \
  --speech-offsets data/final-base/offsets
```

Stage two uses a new output directory and MLflow run. Its update count remains
a separate model-quality decision. Upload its final mix specification, both
cleaning manifests, and all immutable training checkpoints.

## Required artifacts

- both mix specifications and `speech-mix-manifest.json` files;
- four SplitMix64 offset files and metadata JSON files;
- cleaner feature command showing `--disable-foreground`;
- cleaner 10k checkpoint and listening output, plus retained intermediate diagnostic checkpoints;
- compiled cleaner hashes and both Common Voice cleaning manifests;
- original-versus-cleaned listening decision;
- stage-two training metadata, checkpoints, and final listening pack.
