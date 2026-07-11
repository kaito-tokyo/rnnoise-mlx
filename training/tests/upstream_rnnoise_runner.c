#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rnn.h"
#include "rnnoise_data.h"

extern const WeightArray rnnoise_arrays[];

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s features.f32 frames output.f32\n", argv[0]);
        return 2;
    }
    const size_t frames = (size_t)strtoull(argv[2], NULL, 10);
    FILE *input = fopen(argv[1], "rb");
    FILE *output = fopen(argv[3], "wb");
    if (!input || !output) return 1;

    RNNoise model;
    RNNState state;
    memset(&state, 0, sizeof(state));
    if (init_rnnoise(&model, rnnoise_arrays) != 0) return 1;

    for (size_t frame = 0; frame < frames; frame++) {
        float features[65], gains[32], vad;
        if (fread(features, sizeof(float), 65, input) != 65) return 1;
        compute_rnn(&model, &state, gains, &vad, features, 0);
        if (fwrite(gains, sizeof(float), 32, output) != 32 ||
            fwrite(&vad, sizeof(float), 1, output) != 1) return 1;
    }
    fclose(input);
    fclose(output);
    return 0;
}
