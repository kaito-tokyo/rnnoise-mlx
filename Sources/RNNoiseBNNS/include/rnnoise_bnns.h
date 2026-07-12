#ifndef RNNOISE_BNNS_H
#define RNNOISE_BNNS_H

#include <stddef.h>

#define RNNOISE_BNNS_FEATURE_COUNT 65
#define RNNOISE_BNNS_COND_SIZE 128
#define RNNOISE_BNNS_MAX_GRU_SIZE 384
#define RNNOISE_BNNS_GAIN_COUNT 32
#define RNNOISE_BNNS_HISTORY_FRAMES 2

typedef struct RNNoiseBNNSModel RNNoiseBNNSModel;
typedef struct RNNoiseBNNSProcessor RNNoiseBNNSProcessor;

typedef struct {
    size_t frames_seen;
    float feature_history[RNNOISE_BNNS_FEATURE_COUNT * RNNOISE_BNNS_HISTORY_FRAMES];
    float conv1_history[RNNOISE_BNNS_COND_SIZE * RNNOISE_BNNS_HISTORY_FRAMES];
    float gru1[RNNOISE_BNNS_MAX_GRU_SIZE];
    float gru2[RNNOISE_BNNS_MAX_GRU_SIZE];
    float gru3[RNNOISE_BNNS_MAX_GRU_SIZE];
} RNNoiseBNNSFrameState;

typedef struct {
    float gains[RNNOISE_BNNS_GAIN_COUNT];
    float vad;
} RNNoiseBNNSFrameOutput;

enum {
    RNNOISE_BNNS_OK = 0,
    RNNOISE_BNNS_WARMUP = 1,
    RNNOISE_BNNS_INVALID_ARGUMENT = -1,
    RNNOISE_BNNS_IO_ERROR = -2,
    RNNOISE_BNNS_INVALID_MODEL = -3,
    RNNOISE_BNNS_ACCELERATE_ERROR = -4
};

int rnnoise_bnns_model_load(const char *mlmodelc_path, RNNoiseBNNSModel **model);
void rnnoise_bnns_model_destroy(RNNoiseBNNSModel *model);
void rnnoise_bnns_state_init(RNNoiseBNNSFrameState *state);
void rnnoise_bnns_state_reset(RNNoiseBNNSFrameState *state);
int rnnoise_bnns_process(RNNoiseBNNSModel *model,
                         const float features[RNNOISE_BNNS_FEATURE_COUNT],
                         RNNoiseBNNSFrameState *state,
                         RNNoiseBNNSFrameOutput *output);
int rnnoise_bnns_processor_create(const char *mlmodelc_path, RNNoiseBNNSProcessor **processor);
void rnnoise_bnns_processor_destroy(RNNoiseBNNSProcessor *processor);
int rnnoise_bnns_process_audio_frame(RNNoiseBNNSProcessor *processor,
                                     const float input[480], float output[480], float *vad);

#endif
