# MLX test training result — 2026-07-11

The 256-sequence, 320-update procedure in `docs/test-training.md` was run on an
Apple M2 Pro with 32 GB of memory. This model validates the pipeline and is not
intended for audio-quality distribution.

## Data

| split | speech | sequences | bytes |
|---|---|---:|---:|
| train | LibriTTS-R train-clean-100 | 256 | 200,704,000 |
| eval | LibriTTS-R dev-clean | 64 | 50,176,000 |

The dev-clean archive MD5 was `2c1f5312914890634cc2d15783032ff3`, matching
the OpenSLR listing. Noise came from Xiph background noise v2 and foreground
noise; RIRs came from measured_rirs-v2. No existing model weights, inference
outputs, or distillation targets were used.

## Configuration

- MLX 0.32.0, Python 3.14, and Metal
- Batch 8, sequence length 2,000, 10 epochs, and 320 updates
- AdamW, learning rate `1e-3`, betas `(0.8, 0.98)`, gamma `0.25`
- Conv1d 65→128→256, three GRU-256 layers, 32 gains, and one VAD output

## Result

| metric | initial | trained | reloaded |
|---|---:|---:|---:|
| total loss | 0.563442 | 0.185812 | 0.185812 |
| gain loss | 0.562722 | 0.185565 | 0.185565 |
| VAD loss | 0.719916 | 0.247472 | 0.247472 |

- First 32-update mean: 0.401861
- Last 32-update mean: 0.189594, a 52.8% decrease
- Minimum observed training loss: 0.108562
- No NaN or infinity values
- Gain and VAD outputs remained in `[0, 1]`
- Evaluation matched exactly after SafeTensors reload
- Elapsed time: 1,094.1 seconds, or 0.292 updates/s

The measured PyTorch CPU implementation took 57.8 seconds per epoch, or about
578 seconds for ten epochs at 0.553 updates/s. This MLX implementation was about
1.89 times slower, mainly because the GRU was explicitly expanded for 2,000
Python steps. A compiled scan/RNN primitive should be evaluated without changing
the loss or weight format.

## Verdict

The run passed completion, loss improvement, finiteness, output-range, and
save/reload gates. Performance was measured but still required optimization.
