# Reproducible speech-window selection

The current design treats all speech in a split as one concatenated 48 kHz
mono s16le PCM stream. Fixed 20-second RNNoise sequences are selected by start
sample, without requiring clip-level phoneme labels or alignments. Boundaries
between source clips, speakers, or languages may occur inside a sequence.

Phoneme coverage is a property of the population design. Language inventories
come from PHOIBLE and supporting phonological literature; the selector does not
infer or count phonemes in individual clips. A sufficiently large uniform
sample is expected to inherit the population's speech distribution. Language
membership, corpus hours, and any coverage acceptance rule therefore belong in
the population manifest and are checked before expensive feature generation.

Generate one offset manifest per split:

```sh
python -m rnnoise_mlx.tools.select_speech_offsets \
  data/prepared/train_speech.pcm data/offsets/train.txt \
  --count 10000 --seed 141

python -m rnnoise_mlx.tools.select_speech_offsets \
  data/prepared/eval_speech.pcm data/offsets/eval.txt \
  --count 500 --seed 142
```

Each output line is an int16 sample offset. SplitMix64 is initialized once and
consumed exactly once per sequence. A u64 is mapped into the inclusive valid
start range with multiply-high; there is no random-value rejection or
trial-internal resampling. The adjacent `.json` file records the algorithm,
PCM size, sequence length, seed, and RNG call count.

Pass the manifests to feature generation:

```sh
python -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features data/prepared data/features \
  --train-count 10000 --eval-count 500 --speech-offsets data/offsets
```

The vendored `dump_features` reads these speech offsets but retains the RNNoise
feature/target DSP and its existing noise, RIR, gain, and filtering
randomization. The reproducibility contract applies only to speech-window
selection: the same code, PCM bytes, count, and seed produce the same offset
manifest. Downstream augmentation and training randomness are outside it.

The older `select_phoneme_clips` utility remains available for experiments
with clip-level phone annotations, but it is not required by this workflow.
