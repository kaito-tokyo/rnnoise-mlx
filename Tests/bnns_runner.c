/* Native test runner for the Swift/BNNS implementation. */
#include "rnnoise_bnns.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s model.mlmodelc features.f32 frames output.f32\n", argv[0]);
        return 2;
    }
    size_t frames = (size_t)strtoull(argv[3], NULL, 10);
    FILE *input = fopen(argv[2], "rb");
    FILE *output = fopen(argv[4], "wb");
    RNNoiseBNNSModel *model = NULL;
    if (!input || !output || rnnoise_bnns_model_load(argv[1], &model) != RNNOISE_BNNS_OK) return 1;
    RNNoiseBNNSFrameState state;
    rnnoise_bnns_state_init(&state);
    for (size_t frame = 0; frame < frames; frame++) {
        float features[RNNOISE_BNNS_FEATURE_COUNT];
        RNNoiseBNNSFrameOutput result;
        int status;
        if (fread(features, sizeof(float), RNNOISE_BNNS_FEATURE_COUNT, input) != RNNOISE_BNNS_FEATURE_COUNT ||
            ((status = rnnoise_bnns_process(model, features, &state, &result)) != RNNOISE_BNNS_OK && status != RNNOISE_BNNS_WARMUP) ||
            fwrite(result.gains, sizeof(float), RNNOISE_BNNS_GAIN_COUNT, output) != RNNOISE_BNNS_GAIN_COUNT ||
            fwrite(&result.vad, sizeof(float), 1, output) != 1) return 1;
    }
    rnnoise_bnns_model_destroy(model);
    fclose(input); fclose(output);
    return 0;
}
