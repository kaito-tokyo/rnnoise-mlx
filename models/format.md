# Model format v1

The training artifact is a SafeTensors file containing only independently
trained float32 tensors. Its metadata key `config` is JSON with `input_dim`,
`output_dim`, `cond_size`, and `gru_size`.

Tensor names follow the MLX module tree. GRU tensors use gate order
`reset, update, new`:

- `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`
- `gru{1,2,3}.Wx`, `.Wh`, `.b`, `.bhn`
- `gain.weight`, `gain.bias`, `vad.weight`, `vad.bias`

The deploy converter will emit a directory with `manifest.json` plus raw,
little-endian float32 tensors. BNNS must reject unknown format versions or
shape mismatches.

MLX uses `b = [b_ir+b_hr, b_iz+b_hz, b_in]` and `bhn = b_hn`. This is
mathematically equivalent to the separate PyTorch/RNNoise input and recurrent
biases. Gate order remains `reset, update, new`.
