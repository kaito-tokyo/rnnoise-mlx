# Stateful TBPTT experiment — 2026-07-11

> Historical experiment that updated the optimizer after every chunk. The
> current method processes all 2,000 frames with unchanged weights and updates
> once; see [the current experiment](segment-length-100-250-500-experiment.md).

This experiment carried Conv/GRU state between chunks within each 2,000-frame
sequence. It was enabled by `--stateful-tbptt`; the default random-window method
was unchanged.

## State and alignment

The carried state consisted of Conv1 input history `2 × 65`, Conv2 input
history `2 × 128`, and three 256-element GRU hidden states. The first K-frame
chunk produced K-4 outputs; subsequent chunks used history plus K inputs to
produce K outputs. Target ranges were `[3:K-1]` for the first chunk and
`[start-1:end-1]` thereafter, covering 1,996 outputs exactly once. Gradients
were stopped at chunk boundaries.

A 600-frame forward pass split into three 200-frame chunks matched the full
forward pass exactly for gain and VAD. Model structure, tensor shapes,
SafeTensors, and BNNS/C conversion contracts were unchanged.

## Results

| method | chunk | updates | frames | time | frames/s | eval total | gain | VAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random chunk | 200 | 320 | 512,000 | 29.32 s | 17,464 | 0.208147 | 0.207846 | 0.300382 |
| stateful | 200 | 320 | 512,000 | 29.37 s | 17,434 | 0.271462 | 0.271102 | 0.359500 |
| stateful | 800/800/400 | 96 | 512,000 | 82.72 s | 6,190 | 0.363985 | 0.363604 | 0.381121 |
| stateful | 800/800/400 | 320 | 1,708,800 | 263.54 s | 6,484 | 0.232768 | 0.232436 | 0.332699 |
| random chunk | 800 | 320 | 2,048,000 | 311.14 s | 6,582 | 0.195729 | 0.195468 | 0.261367 |

All results were finite, outputs stayed in `[0, 1]`, and evaluation matched
after SafeTensors reload.

## Interpretation

Although state and alignment matched full forward execution, updating after
every chunk did not improve accuracy. Likely causes include correlated
consecutive chunks, passing state computed by pre-update weights into an
updated model, greater diversity in random-window training, and imperfectly
matched updates or frames in the length-800 comparison.

Stateful per-chunk updates should not be the default. The next comparison
should accumulate gradients across multiple chunks while keeping weights fixed,
then update once per sequence or group of chunks. Compare against random lengths
200 and 800 with matched frames, matched updates, and fixed-WAV listening tests.
