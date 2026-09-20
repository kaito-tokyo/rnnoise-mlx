"""RNNoise gain/VAD objective, separate from the sequence model."""


def rnnoise_loss(predicted_gain, predicted_vad, target_gain, target_vad, *, gamma):
    target = target_gain.clamp_min(0)
    target = target * (8 * target).tanh().square()
    active = (target_gain + 1).clamp_max(1)
    error = predicted_gain.pow(gamma) - target.pow(gamma)
    gain_loss = ((1 + 5 * target_vad) * active * error.square()).mean()
    vad_weight = (2 * target_vad - 1).abs()
    vad_positive_loss = -target_vad * (0.01 + predicted_vad).log()
    vad_negative_loss = -(1 - target_vad) * (1.01 - predicted_vad).log()
    vad_loss = (vad_weight * (vad_positive_loss + vad_negative_loss)).mean()
    return gain_loss + 0.001 * vad_loss
