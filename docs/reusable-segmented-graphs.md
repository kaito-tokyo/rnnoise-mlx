# Reusable segmented training graphs — 2026-07-13

## Design

Segmented TBPTT no longer compiles all 2,000 frames as one expanded graph.
Training now compiles and reuses separate `first_chunk` and `next_chunk`
forward/backward graphs. Conv and GRU state is carried between calls with the
gradient stopped at each boundary. Chunk gradients are weighted by their valid
output-frame count, accumulated on the device, and passed to AdamW once after
the complete 2,000-frame sequence.

The reset comparison remains supported with a reusable first-chunk graph for
every segment. The optimizer and update-based learning-rate schedule still
advance once per full sequence.

An asynchronous materialization is submitted after every chunk. Removing it
increased the length-100 peak from about 1.01 GB to 4.84 GB and reduced
throughput, because the lazy graph grew across reusable calls.

## Numerical validation

Tests compare the reusable implementation with the former monolithic
objective for:

- state carry with every valid output
- state carry with reset-equivalent target exclusion
- state reset
- two compiled optimizer updates with the same `next_chunk` graph called more
  than once per update

Loss, model parameters, and AdamW state match within `2e-6` absolute and
relative tolerance. The complete test suite passes: `44 passed`.

## Synthetic Metal benchmark

Measured on the same Apple M2 Pro class of machine with MLX 0.32.0. Each update
uses a 2,000-frame sequence; reported memory is the Metal peak after one warmup
update.

| segment | batch | seconds/update | frames/s | peak memory |
|---:|---:|---:|---:|---:|
| 100 | 8 | 1.025 | 15,603 | 1.01 GB |
| 500 | 8 | 1.870 | 8,557 | 11.35 GB |
| 100 | 16 | 1.227 | 26,077 | 1.47 GB |

The historical monolithic batch-8 measurements were 19,558 frames/s at length
100 and 8,625 frames/s at length 500, with about 25 GB allocated during the
observed length-500 run. Reusable length 500 is within about 1% of the
historical throughput while reducing observed memory substantially. Length 100
at batch 8 is about 20% slower, but batch 16 exceeds the historical batch-8
throughput by about 33% while remaining below 1.5 GB in this synthetic
measurement.

The 11.35 GB length-500 peak was unchanged when every optimizer update was
explicitly submitted for asynchronous evaluation. It is therefore attributed
to the single 500-frame forward/backward graph rather than lazy accumulation
between optimizer updates. The superlinear increase from length 100 reflects
the Python-expanded three-layer GRU backward graph and its temporary storage;
it is the main target for fused-GRU or scan work.

These timings are short synthetic measurements and should be confirmed with
the fixed training corpus before changing production batch size or learning
hyperparameters. The next compute optimization is a fused GRU or scan inside
each reusable chunk graph.

## Stability recheck

A length-500, batch-8 stress check ran 16 consecutive compiled optimizer
updates after warmup. Every loss was finite. Resetting the Metal peak counter
before each update reported 11,348,231,004–11,348,231,048 bytes, a spread of
only 44 bytes. Active memory after each synchronized update remained between
40,911,100 and 40,911,113 bytes. This shows no update-to-update graph or active
memory growth in the tested run.
