# SPDX-FileCopyrightText: 2026 Kaito Udagawa <umireon@kaito.tokyo>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelConfig:
    """Describe the RNNoise network dimensions shared by all backends."""

    input_dim: int = field(default=65, init=False)
    output_dim: int = field(default=32, init=False)
    cond_size: int = field(default=128, init=False)
    gru_size: int = field(default=384, init=False)
