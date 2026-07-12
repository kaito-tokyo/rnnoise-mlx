# Roadmap and acceptance gates

1. MLX smoke training: 320 updates, finite loss, and a downward loss trend.
2. Held-out features: report gain loss, VAD loss, and inference throughput.
3. Freeze model format and verify sequence-vs-frame numerical parity.
4. Implement the BNNS graph and compare its outputs with MLX within tolerance.
5. Connect the pinned RNNoise C DSP and measure end-to-end real-time factor.
6. Train for thousands to 10,000+ updates and run objective and listening tests.

A 320-update checkpoint proves pipeline viability only; it is not a
release-quality model.

## Training optimization order

1. Use segment length 100 for corpus additions and rapid subjective screening.
2. Generate WAV files from a fixed evaluation set and run user A/B tests.
3. Retrain only promising data configurations with segment length 500 and state carry.
4. Carry convolution history and all three GRU states; stop gradients only at boundaries.
5. Process four equally weighted 500-frame segments and update the optimizer once.
6. Run thousands to 10,000+ updates and select using both evaluation loss and WAV quality.
7. Compare batch size, learning rate, and gradient clipping only when needed.

In equal-frame and equal-update tests of lengths 100, 250, and 500, length 500
had the best evaluation loss. Carrying state at length 500 improved total loss
by about 1.90% over reset, and the paired bootstrap interval over 64 evaluation
sequences excluded zero. Length 100 was about 2.27 times faster, so it remains
the development default. See
[the experiment](segment-length-100-250-500-experiment.md).

The next speed experiment should reuse a compiled 500-frame graph four times,
accumulate gradients on device, and update once. The monolithic graph reached
96% GPU use and about 25 GB of allocated GPU memory on an M2 Pro, so this should
first be validated with a short A/B benchmark.

Mixed precision is deferred because it may be counterproductive for this small
model. Metal profiling and an MLX fused-GRU primitive are later candidates.
