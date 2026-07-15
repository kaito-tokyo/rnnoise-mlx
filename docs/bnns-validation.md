# C/BNNSGraph inference validation — 2026-07-12

The base-model SafeTensors checkpoint is converted directly to a Core ML
package and compiled `.mlmodelc`.
The C implementation loads that model with `BNNSGraphCompileFromFile`, executes
the complete frame network in one graph context, and preserves causal
convolution and recurrent state across calls.

## MLX numerical parity

`python -m rnnoise_mlx.tools.verify_bnns` compared 100 held-out frames with MLX after
discarding the four-frame convolution warm-up.

```text
max absolute error:  3.57627869e-07
mean absolute error: 3.83831740e-08
tolerance:           1e-05
```

## Audio-path validation

A held-out LibriTTS-R test utterance was mixed with a held-out MUSAN noise file
at 5 dB SNR. Output length was preserved and all output samples were finite.

| signal | SI-SDR |
|---|---:|
| noisy | 4.9684 dB |
| C DSP + BNNS output | 5.6077 dB |
| improvement | +0.6393 dB |

This is a pipeline acceptance check, not a broad quality claim. A larger fixed
evaluation suite and listening tests remain necessary for release decisions.

## Format validation

AVFoundation successfully decoded both a 24 kHz mono WAV and a 48 kHz stereo
AAC/M4A excerpt from an LDTX recording. The executable generated 48 kHz mono
WAV files with exactly the expected duration (10.120 seconds and 5.000 seconds).

The historical source weight bundle used for this validation is stored at:

```text
/Users/umireon/Datasets/base-model-320/model.bnns
```

It is a legacy `RNMLXBN1` intermediate artifact retained as validation
provenance. New conversions accept training or canonical SafeTensors directly.
Deploy the compiled `RNNoiseGraph.mlmodelc` directory with the application.
