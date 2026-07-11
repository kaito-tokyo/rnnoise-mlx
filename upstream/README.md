# Upstream RNNoise

DSP integration is pinned to fork revision
`70f1d256acd4b34a572f999a05c87bf00b67730d` from
`https://github.com/kaito-tokyo/rnnoise.git`.

The fork remains responsible for minimal upstream patches, including the
65536-point RIR FFT `fstride` bound and moving large RIR FFT buffers off the
stack. This repository will consume the pinned fork as a submodule once the
DSP build target is introduced; neural weights from that repository are not
part of release artifacts.

