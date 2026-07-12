import mlx.core as mx


def rnnoise_loss(pred_gain, pred_vad, gain, vad, gamma: float = 0.25):
    gain = gain[:, 3:-1, :]
    vad = vad[:, 3:-1, :]
    return rnnoise_loss_aligned(pred_gain, pred_vad, gain, vad, gamma)


def rnnoise_loss_aligned(pred_gain, pred_vad, gain, vad, gamma: float = 0.25):
    target = mx.maximum(gain, 0)
    target = target * mx.square(mx.tanh(8 * target))
    active = mx.minimum(gain + 1, 1)
    error = pred_gain**gamma - target**gamma
    gain_loss = mx.mean((1 + 5 * vad) * active * mx.square(error))
    vad_loss = mx.mean(
        mx.abs(2 * vad - 1)
        * (-vad * mx.log(0.01 + pred_vad) - (1 - vad) * mx.log(1.01 - pred_vad))
    )
    return gain_loss + 0.001 * vad_loss, gain_loss, vad_loss
