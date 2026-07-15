# Model format v1

The training artifact is a SafeTensors file containing only independently
trained float32 tensors. Its metadata key `config` is JSON with `input_dim`,
`output_dim`, `cond_size`, and `gru_size`.

Tensor names follow the MLX module tree. GRU tensors use gate order
`reset, update, new`:

- `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`
- `gru{1,2,3}.Wx`, `.Wh`, `.b`, `.bhn`
- `gain.weight`, `gain.bias`, `vad.weight`, `vad.bias`

The canonical deployment schema uses the same 20 float32 tensors in a standard
SafeTensors container. Metadata identifies `rnnoise-mlx-canonical` schema
version 1, the model dimensions, GRU gate order, and bias layout. Both the
original `65/128/256/32` MLX profile and the official `65/128/384/32` RNNoise
profile are supported. Training artifacts already have this tensor layout and
can be passed directly to `python -m rnnoise_mlx.tools.export_bnns_graph`.
`python -m rnnoise_mlx.tools.export_canonical_weights` optionally writes an explicitly
versioned canonical artifact for interchange. The graph converter validates the weights and emits a Core ML
package containing one stateful frame graph. Xcode's `coremlcompiler` turns that
package into the `.mlmodelc` directory loaded by the C BNNSGraph runtime.

The graph has explicit inputs and outputs for the two causal-convolution
histories and three GRU hidden states. Keeping state at the API boundary makes
reset behavior deterministic and lets the C runtime execute without Classic
BNNS filters.

`python -m rnnoise_mlx.tools.convert_official_weights` imports official PyTorch
checkpoints into this canonical representation without retraining. PyTorch
Conv1d weights are transposed into time-major canonical order. GRU input and
recurrent reset/update biases are summed, while the candidate recurrent bias is
stored separately as `bhn`; this preserves the GRU equation even though the
serialized bias split changes. The reverse conversion emits a checkpoint that
upstream's `dump_rnnoise_weights.py` accepts.

Legacy files with the `RNMLXBN1` magic remain readable for migration, but new
canonical artifacts are written as SafeTensors.

MLX uses `b = [b_ir+b_hr, b_iz+b_hz, b_in]` and `bhn = b_hn`. This is
mathematically equivalent to the separate PyTorch/RNNoise input and recurrent
biases. Gate order remains `reset, update, new`.
