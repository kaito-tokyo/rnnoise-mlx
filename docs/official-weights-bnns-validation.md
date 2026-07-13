# Official RNNoise weights with BNNSGraph — 2026-07-12

The official `rnnoise10Ga_12.pth` and `rnnoise10Gb_15.pth` checkpoints were
converted without retraining. The generated `rnnoise_data.c` used by the
upstream C build corresponds most closely to the Ga checkpoint, so Ga is used
for the direct C-versus-BNNS comparison below.

```text
official PyTorch checkpoint
  -> canonical RNMLXBN1 float32 bundle
  -> Core ML package
  -> compiled BNNSGraph mlmodelc
```

The canonical profile preserves the official 128-channel first convolution,
384-channel second convolution, three 384-unit GRUs, 32 gain outputs, and one
VAD output. Exporting the canonical bundle back to a PyTorch checkpoint
preserved inference within `1e-6`, and upstream `dump_rnnoise_weights.py`
successfully generated C weights from the round-tripped checkpoint.

## Numerical validation

Two thousand held-out feature frames were evaluated with streaming zero-history
semantics. BNNSGraph was compared with a canonical NumPy float32 reference:

```text
maximum absolute error: 5.81145287e-06
mean absolute error:    1.54611314e-07
correlation:             1.0
```

The upstream C `compute_rnn()` path and the Ga BNNSGraph output had overall
correlation `0.988814342`, gain correlation `0.988549677`, and VAD correlation
`0.981003622`. Their mean absolute difference was `0.0148032168`. The C build
uses generated quantized/sparse weights, while BNNSGraph uses the checkpoint's
dense float32 weights, so exact agreement is not expected.

For comparison, the Gb checkpoint only reached correlation `0.83999827` with
the same generated C model, confirming that Ga is the appropriate checkpoint
for this direct comparison.

## LDTX audio validation

The complete 4,756.139-second `side-track-2` microphone recording was processed
with the converted official weights. The output contains 228,294,656 finite
48-kHz mono samples, matching the input duration.

Over a 60-second excerpt, the official upstream C output and the Ga BNNSGraph
output had waveform correlation `0.999627744` after compensating for a
480-sample alignment difference. The waveform RMSE was `0.000399026` for an
upstream signal RMS of `0.013721607`. Their whole-file levels were also close:

| implementation | mean level | peak level |
|---|---:|---:|
| upstream official C | -34.3 dB | -0.1 dB |
| official Ga weights with BNNSGraph | -34.2 dB | 0.0 dB |

The upstream C model is quantized/sparse, while this BNNSGraph conversion uses
the checkpoint's dense float32 weights, so bit-exact PCM is not expected.

## Reproducing the direct C comparison

Build `Tests/upstream_rnnoise_runner.c` against the prepared upstream
static library, then pass the resulting executable to
`verify_official_bnns.py --upstream-runner`. The runner invokes upstream
`compute_rnn()` directly and writes 32 gains plus one VAD value per frame.
