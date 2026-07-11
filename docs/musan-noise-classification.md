# MUSAN noise classification summary

## Purpose

MUSAN's `noise/` directory mixes long ambient recordings, short transients,
technical tones, and other sound effects. RNNoise training accepts separate
background and foreground streams, so the files are ranked by temporal
stationarity and transient activity before manual review.

This classifier is a corpus triage tool. Its scores are rankings, not
probabilities, and its labels do not establish semantic content or licensing.
The original WAV files and MUSAN metadata remain authoritative.

## Input

- Corpus: MUSAN `noise/`
- Files analyzed: 930 WAV files
- Total duration measured: 6.23 hours
- Sources present in MUSAN: `free-sound/` and `sound-bible/`
- Audio is converted to mono for analysis by averaging channels. The original
  files are not modified.
- Supported PCM sample widths: 16, 24, and 32 bit.

## Analysis window

Each file is analyzed using:

- frame length: 20 ms
- hop length: 10 ms
- Hann window before FFT
- magnitude spectra normalized by each frame's L2 norm

Normalization makes spectral-change measurements less dependent on absolute
recording gain.

## Recorded metrics

The CSV contains one row per WAV and these metrics:

| field | meaning |
|---|---|
| `duration_s` | file duration |
| `rms_dbfs` | mean short-time RMS in dBFS |
| `rms_cv` | coefficient of variation of short-time RMS |
| `spectral_flux_median` | typical positive frame-to-frame spectral change |
| `spectral_flux_p95` | 95th percentile spectral change |
| `spectral_flux_peak` | maximum spectral change |
| `onset_density_hz` | onsets per second above median flux + 6 MAD |
| `crest_factor` | peak amplitude divided by whole-file RMS |
| `silence_ratio` | fraction of frames below -60 dBFS RMS |
| `clipping_ratio` | fraction of samples at or above 0.999 full scale |
| `stationarity_score` | combined background ranking score |
| `transient_score` | combined foreground ranking score |

MAD means median absolute deviation.

## Stationarity score

Background candidates should have stable short-time energy and relatively
small typical spectral changes. The score is:

```text
stationarity_score =
  1 / (1 + 2.5*rms_cv + 80*spectral_flux_median + 0.5*silence_ratio)
```

A larger value means more stationary. Duration is checked separately so that a
very short, steady tone is not automatically treated as background ambience.

## Transient score

Foreground candidates should exhibit sharp changes, high peaks relative to
RMS, or frequent onsets. The score is:

```text
transient_score =
  log(1 + 8*crest_factor)
  * (120*spectral_flux_p95 + 40*spectral_flux_peak + onset_density_hz)
```

A larger value means stronger or more frequent transient behavior.

## Current decision rules

The current MUSAN-calibrated defaults are:

```text
background candidate:
  duration >= 10 seconds
  stationarity_score >= 0.23

foreground candidate:
  transient_score >= 45.0

forced review:
  clipping_ratio > 0.001
```

Labels are assigned as follows:

| background condition | foreground condition | label |
|---|---|---|
| true | false | `background` |
| false | true | `foreground` |
| true | true | `mixed` |
| false | false | `review` |

Files with excessive clipping are labeled `review` regardless of their scores.
Unreadable or unsupported WAV files are labeled `error` with the exception
message stored in the output.

## Calibration basis

The first generic thresholds did not match the MUSAN score distribution. The
current values were selected after measuring all 930 files and comparing them
with MUSAN's `free-sound/ANNOTATIONS`, which lists 37 files as background
noise.

For those 37 annotated background files:

- median stationarity score: approximately 0.236
- median transient score: approximately 30.9
- the conservative automatic rules recovered 17 as `background`
- 19 remained `review`
- 1 was marked `foreground`

This intentionally favors precision over recall. An annotated ambient sound
can be non-stationary (for example rain, wind, crowds, or a changing street
scene) and remain in `review`; that does not mean the MUSAN annotation is
wrong.

## Current result

| label | files | duration |
|---|---:|---:|
| `background` | 100 | 1.277 h |
| `foreground` | 223 | 1.226 h |
| `mixed` | 19 | 0.345 h |
| `review` | 588 | 3.379 h |

Only `background` and `foreground` are included in the current automatic
baseline. `mixed` and `review` remain excluded until manual inspection.

## Reproduction

Run the classifier from the repository root:

```sh
.venv/bin/python scripts/classify_noise.py \
  data/corpora/musan/musan/noise \
  data/corpora/musan/noise-classification.csv \
  --background-threshold 0.23 \
  --foreground-threshold 45 \
  --min-background-seconds 10 \
  --organize-dir data/corpora/musan/review
```

The organizer creates relative symbolic links and does not duplicate or modify
the WAV files. Review folders are:

```text
data/corpora/musan/review/background/
data/corpora/musan/review/foreground/
data/corpora/musan/review/mixed/
data/corpora/musan/review/review/
```

The complete metric output is:

```text
data/corpora/musan/noise-classification.csv
```

## Train/eval split

After classification, accepted files are split by hashing the salt and relative
path with SHA-256. The default eval fraction is 15% and the default salt is
`rnnoise-mlx-musan-v1`. This makes the split deterministic and file-disjoint.

Current split:

| split | background | foreground |
|---|---:|---:|
| train | 82 | 194 |
| eval | 18 | 29 |

There is no file overlap between train and eval. The manifest is generated by:

```sh
.venv/bin/python scripts/split_noise_manifest.py \
  data/corpora/musan/noise-classification.csv \
  data/training/musan-baseline/split-manifest.csv
```

## Limitations and manual review

The classifier does not identify the sound source. It cannot reliably detect:

- intelligible speech, music, television, or privacy-sensitive content
- whether a transient belongs inside a longer background recording
- perceptual suitability for RNNoise
- duplicated or near-duplicated recordings
- source-level leakage when different files originate from the same recording
  session

Before promoting additional files, listen to the score boundaries and inspect
waveforms. In particular:

1. review MUSAN's 37 annotated background files first;
2. inspect `background` files just above stationarity 0.23;
3. inspect `foreground` files just above transient 45;
4. separate or reject `mixed` files rather than blindly concatenating them;
5. exclude intelligible speech and music from the noise corpus;
6. keep source/session groups together if provenance reveals related files.

Threshold changes require regenerating the classification CSV, review links,
split manifest, PCM files, and all downstream RNNoise feature files.
