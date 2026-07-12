# Base-model dataset provenance

Prepared on 2026-07-12 with seed 141 from archives in
`/Users/umireon/Datasets`. The base-model inputs use public data only:
LibriTTS-R train-clean-100/360, MUSAN, and simulated RIRs from RIRS_NOISES.
Xiph noise and self-recorded audio are excluded.

## Source archives

| Archive | SHA-256 |
| --- | --- |
| `train_clean_100.tar.gz` | `6f0cef20fb1e72928f21f00711d8cda7acb2049fd1cef78c98e0c2db77de2547` |
| `train_clean_360.tar.gz` | `9ac11d2421b213efa29aa684d8b5373a2548afcc5c1e655d20f793eb1df728a5` |
| `dev_clean.tar.gz` | `02ea52de47ca670d7ad86e714a6c58b207660a3f8a371ad102156776d2262a0c` |
| `test_clean.tar.gz` | `d4a2a7cdeeb68a6cfd628559739879bcb29556364cc13f38413930d59369a7f1` |
| `musan.tar.gz` | `86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1` |
| `rirs_noises.zip` | `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` |

## Prepared inputs

- Speech: 149,694 train, 5,736 development-eval, 4,837 final-test files.
- Background: 1,431 train and 159 eval files.
- Foreground: 377 train and 49 eval files.
- Simulated RIR pool: 54,196 train and 5,804 eval files; rendering uses
  the first 512 entries from each deterministic manifest.
- Rendered input: the first 40,000 train speech files, all 5,736 eval speech
  files, all available noise files, and 512 RIRs per split.
- Generated features: 10,000 train sequences (`7,840,000,000` bytes) and
  500 eval sequences (`392,000,000` bytes), with 2,000 frames and 98 float32
  values per frame.

The ignored local artifacts live under `data/`. Preserve
`data/manifests/` together with the eventual model artifacts.
