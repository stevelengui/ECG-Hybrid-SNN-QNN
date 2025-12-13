#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

// Buffer structure for hybrid model
typedef struct {
    // Input buffer
    int8_t input_buf[INPUT_SIZE];
    
    // Spatial path (QNN)
    int32_t conv1_out[CONV1_FILTERS * CONV1_OUT_SIZE];
    int32_t conv2_out[CONV2_FILTERS * CONV2_OUT_SIZE];
    
    // Temporal path (SNN)
    int8_t snn_input[SNN_INPUT_SIZE * TIME_STEPS];
    int8_t spike_train[TIME_STEPS * SNN_HIDDEN];
    int8_t lstm_state[LSTM_HIDDEN];
    
    // Feature router
    int8_t router_output[2];
    
    // Fusion
    int32_t fused_features[FUSION_SIZE];
    int32_t fc1_out[FC1_SIZE];
    
    // Output
    int32_t output[OUTPUT_SIZE];
    
    // Thermal management
    uint8_t precision_mode;
    int16_t temperature;
    
    // Benchmarking
    uint32_t total_cycles;
    uint32_t inference_count;
} HybridModelBuffers;

// Function prototypes
void model_init(HybridModelBuffers* buffers);
void model_predict(HybridModelBuffers* buffers, const int8_t* input, uint8_t domain);

// Benchmarking functions
void benchmark_latency(HybridModelBuffers* buffers, uint8_t domain, uint32_t iterations);
float calculate_tops(uint32_t cycles, uint32_t operations, uint32_t cpu_freq_hz);
float calculate_tops_w(float tops, uint32_t power_mw);

#endif // MODEL_H
