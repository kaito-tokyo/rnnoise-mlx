#include "rnnoise_bnns.h"

#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "rnnoise.h"

/* The macOS 15 SDK exposes this public symbol in Accelerate, although older
 * header revisions accidentally hide its declaration behind an availability
 * guard. Keep the declaration local until the minimum supported SDK does. */
extern bnns_graph_t RNNoiseBNNSGraphCompileFromFile(
    const char *filename, const char *function, bnns_graph_compile_options_t options)
    __asm__("_BNNSGraphCompileFromFile_v2");

enum {
    ARG_GAINS, ARG_VAD, ARG_FEATURE_HISTORY_OUT, ARG_CONV1_HISTORY_OUT,
    ARG_GRU1_OUT, ARG_GRU2_OUT, ARG_GRU3_OUT, ARG_FEATURES,
    ARG_FEATURE_HISTORY_IN, ARG_CONV1_HISTORY_IN, ARG_GRU1_IN, ARG_GRU2_IN,
    ARG_GRU3_IN, ARG_COUNT
};

struct RNNoiseBNNSModel {
    bnns_graph_t graph;
    bnns_graph_context_t context;
    size_t positions[ARG_COUNT];
    void *workspace;
    size_t workspace_size;
    size_t gru_size;
};

struct RNNoiseBNNSProcessor {
    RNNoiseBNNSModel *model;
    RNNoiseBNNSFrameState neural_state;
    DenoiseState *dsp_state;
};

static const char *const argument_names[ARG_COUNT] = {
    "gains", "vad", "next_feature_history", "next_conv1_history",
    "next_gru1_state", "next_gru2_state", "next_gru3_state", "features",
    "feature_history", "conv1_history", "gru1_state", "gru2_state", "gru3_state"
};

static int allocate_workspace(RNNoiseBNNSModel *model) {
    model->workspace_size = BNNSGraphContextGetWorkspaceSize(model->context, NULL);
    if (model->workspace_size == SIZE_MAX) return RNNOISE_BNNS_ACCELERATE_ERROR;
    if (model->workspace_size == 0) return RNNOISE_BNNS_OK;
    size_t page_size = (size_t)getpagesize();
    if (posix_memalign(&model->workspace, page_size, model->workspace_size) != 0)
        return RNNOISE_BNNS_IO_ERROR;
    return RNNOISE_BNNS_OK;
}

int rnnoise_bnns_model_load(const char *path, RNNoiseBNNSModel **out) {
    if (!path || !out) return RNNOISE_BNNS_INVALID_ARGUMENT;
    *out = NULL;
    RNNoiseBNNSModel *model = calloc(1, sizeof(*model));
    if (!model) return RNNOISE_BNNS_IO_ERROR;
    bnns_graph_compile_options_t options = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetOptimizationPreference(
        options, BNNSGraphOptimizationPreferencePerformance);
    model->graph = RNNoiseBNNSGraphCompileFromFile(path, NULL, options);
    BNNSGraphCompileOptionsDestroy(options);
    if (!model->graph.data || model->graph.size == 0) {
        rnnoise_bnns_model_destroy(model);
        return RNNOISE_BNNS_INVALID_MODEL;
    }
    if (BNNSGraphGetArgumentCount(model->graph, NULL) != ARG_COUNT) {
        rnnoise_bnns_model_destroy(model);
        return RNNOISE_BNNS_INVALID_MODEL;
    }
    for (size_t i = 0; i < ARG_COUNT; i++) {
        model->positions[i] = BNNSGraphGetArgumentPosition(model->graph, NULL, argument_names[i]);
        if (model->positions[i] == SIZE_MAX) {
            rnnoise_bnns_model_destroy(model);
            return RNNOISE_BNNS_INVALID_MODEL;
        }
    }
    model->context = BNNSGraphContextMake(model->graph);
    BNNSTensor state_tensor = {0};
    if (!model->context.data ||
        BNNSGraphContextGetTensor(model->context, NULL, "gru1_state", true, &state_tensor) != 0 ||
        state_tensor.rank != 1 || state_tensor.shape[0] <= 0 ||
        state_tensor.shape[0] > RNNOISE_BNNS_MAX_GRU_SIZE) {
        rnnoise_bnns_model_destroy(model);
        return RNNOISE_BNNS_INVALID_MODEL;
    }
    model->gru_size = (size_t)state_tensor.shape[0];
    if (allocate_workspace(model) != RNNOISE_BNNS_OK) {
        rnnoise_bnns_model_destroy(model);
        return RNNOISE_BNNS_ACCELERATE_ERROR;
    }
    *out = model;
    return RNNOISE_BNNS_OK;
}

void rnnoise_bnns_model_destroy(RNNoiseBNNSModel *model) {
    if (!model) return;
    free(model->workspace);
    if (model->context.data) BNNSGraphContextDestroy(model->context);
    free(model->graph.data);
    free(model);
}

void rnnoise_bnns_state_init(RNNoiseBNNSFrameState *state) {
    if (state) memset(state, 0, sizeof(*state));
}

void rnnoise_bnns_state_reset(RNNoiseBNNSFrameState *state) {
    rnnoise_bnns_state_init(state);
}

static void set_argument(bnns_graph_argument_t arguments[ARG_COUNT], size_t position,
                         void *data, size_t size) {
    arguments[position].data_ptr = data;
    arguments[position].data_ptr_size = size;
}

int rnnoise_bnns_process(RNNoiseBNNSModel *model, const float features[65],
                         RNNoiseBNNSFrameState *state, RNNoiseBNNSFrameOutput *output) {
    if (!model || !features || !state || !output) return RNNOISE_BNNS_INVALID_ARGUMENT;
    float next_feature_history[130], next_conv1_history[256];
    float next_gru1[RNNOISE_BNNS_MAX_GRU_SIZE];
    float next_gru2[RNNOISE_BNNS_MAX_GRU_SIZE];
    float next_gru3[RNNOISE_BNNS_MAX_GRU_SIZE];
    size_t gru_bytes = model->gru_size * sizeof(float);
    bnns_graph_argument_t arguments[ARG_COUNT] = {0};
    set_argument(arguments, model->positions[ARG_GAINS], output->gains, sizeof(output->gains));
    set_argument(arguments, model->positions[ARG_VAD], &output->vad, sizeof(output->vad));
    set_argument(arguments, model->positions[ARG_FEATURE_HISTORY_OUT], next_feature_history, sizeof(next_feature_history));
    set_argument(arguments, model->positions[ARG_CONV1_HISTORY_OUT], next_conv1_history, sizeof(next_conv1_history));
    set_argument(arguments, model->positions[ARG_GRU1_OUT], next_gru1, gru_bytes);
    set_argument(arguments, model->positions[ARG_GRU2_OUT], next_gru2, gru_bytes);
    set_argument(arguments, model->positions[ARG_GRU3_OUT], next_gru3, gru_bytes);
    set_argument(arguments, model->positions[ARG_FEATURES], (void *)features, 65 * sizeof(float));
    set_argument(arguments, model->positions[ARG_FEATURE_HISTORY_IN], state->feature_history, sizeof(state->feature_history));
    set_argument(arguments, model->positions[ARG_CONV1_HISTORY_IN], state->conv1_history, sizeof(state->conv1_history));
    set_argument(arguments, model->positions[ARG_GRU1_IN], state->gru1, gru_bytes);
    set_argument(arguments, model->positions[ARG_GRU2_IN], state->gru2, gru_bytes);
    set_argument(arguments, model->positions[ARG_GRU3_IN], state->gru3, gru_bytes);
    if (BNNSGraphContextExecute(model->context, NULL, ARG_COUNT, arguments,
                                model->workspace_size, model->workspace) != 0)
        return RNNOISE_BNNS_ACCELERATE_ERROR;
    memcpy(state->feature_history, next_feature_history, sizeof(next_feature_history));
    memcpy(state->conv1_history, next_conv1_history, sizeof(next_conv1_history));
    state->frames_seen++;
    /* Legacy 256-unit MLX checkpoints were trained only on valid convolution
     * outputs. Official-compatible 384-unit models use upstream's zero-state
     * streaming boundary behavior from the first frame. */
    if (model->gru_size == 256 && state->frames_seen <= 4) {
        memset(output, 0, sizeof(*output));
        return RNNOISE_BNNS_WARMUP;
    }
    memcpy(state->gru1, next_gru1, gru_bytes);
    memcpy(state->gru2, next_gru2, gru_bytes);
    memcpy(state->gru3, next_gru3, gru_bytes);
    return RNNOISE_BNNS_OK;
}

static int gain_callback(void *context, const float features[65], float gains[32], float *vad) {
    RNNoiseBNNSProcessor *processor = context;
    RNNoiseBNNSFrameOutput output;
    int status = rnnoise_bnns_process(processor->model, features, &processor->neural_state, &output);
    if (status == RNNOISE_BNNS_WARMUP) {
        for (size_t i = 0; i < 32; i++) gains[i] = 1.f;
        if (vad) *vad = 0.f;
        return 0;
    }
    if (status != RNNOISE_BNNS_OK) return -1;
    memcpy(gains, output.gains, sizeof(output.gains));
    if (vad) *vad = output.vad;
    return 0;
}

int rnnoise_bnns_processor_create(const char *path, RNNoiseBNNSProcessor **out) {
    if (!path || !out) return RNNOISE_BNNS_INVALID_ARGUMENT;
    *out = NULL;
    RNNoiseBNNSProcessor *processor = calloc(1, sizeof(*processor));
    if (!processor) return RNNOISE_BNNS_IO_ERROR;
    int status = rnnoise_bnns_model_load(path, &processor->model);
    if (status != RNNOISE_BNNS_OK) { free(processor); return status; }
    rnnoise_bnns_state_init(&processor->neural_state);
    processor->dsp_state = rnnoise_create(NULL);
    if (!processor->dsp_state) { rnnoise_bnns_processor_destroy(processor); return RNNOISE_BNNS_IO_ERROR; }
    *out = processor;
    return RNNOISE_BNNS_OK;
}

void rnnoise_bnns_processor_destroy(RNNoiseBNNSProcessor *processor) {
    if (!processor) return;
    rnnoise_destroy(processor->dsp_state);
    rnnoise_bnns_model_destroy(processor->model);
    free(processor);
}

int rnnoise_bnns_process_audio_frame(RNNoiseBNNSProcessor *processor,
                                     const float input[480], float output[480], float *vad) {
    if (!processor || !input || !output) return RNNOISE_BNNS_INVALID_ARGUMENT;
    return rnnoise_process_frame_with_callback(processor->dsp_state, output, input,
                                                gain_callback, processor, vad) == 0
        ? RNNOISE_BNNS_OK : RNNOISE_BNNS_ACCELERATE_ERROR;
}
