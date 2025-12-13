#include "firmware/model/model_weights.h"
#include "ops/math_ops.h"
#include "utils/memutils.h"
#include "utils/cycle_count.h"

// Enlever les variables inutilisées ou les commenter
// #ifdef ENABLE_BENCHMARKING
// static uint32_t total_inference_time_us = 0;
// static uint32_t inference_count_total = 0;
// #endif

void model_init(HybridModelBuffers* buffers) {
    // Initialize all buffers to zero
    memset(buffers->input_buf, 0, INPUT_SIZE);
    memset(buffers->conv1_out, 0, CONV1_FILTERS * CONV1_OUT_SIZE * sizeof(int32_t));
    memset(buffers->conv2_out, 0, CONV2_FILTERS * CONV2_OUT_SIZE * sizeof(int32_t));
    memset(buffers->snn_input, 0, SNN_INPUT_SIZE * TIME_STEPS);
    memset(buffers->spike_train, 0, TIME_STEPS * SNN_HIDDEN);
    memset(buffers->lstm_state, 0, LSTM_HIDDEN);
    memset(buffers->router_output, 0, 2);
    memset(buffers->fused_features, 0, FUSION_SIZE * sizeof(int32_t));
    memset(buffers->fc1_out, 0, FC1_SIZE * sizeof(int32_t));
    memset(buffers->output, 0, OUTPUT_SIZE * sizeof(int32_t));
    
    // Initialize thermal management
    buffers->precision_mode = PRECISION_HIGH;
    buffers->temperature = 25;
    buffers->total_cycles = 0;
    buffers->inference_count = 0;
}

void model_predict(HybridModelBuffers* buffers, const int8_t* input, uint8_t domain) {
    (void)domain;  // Mark as unused to avoid warning
    
    // Start cycle counting for latency measurement
    #ifdef ENABLE_BENCHMARKING
    cycle_t start_cycles = get_cycle_count();
    #endif
    
    // 1. Apply thermal management
    thermal_management(buffers);
    
    // 2. Copy input
    for (int i = 0; i < INPUT_SIZE; i++) {
        buffers->input_buf[i] = input[i];
    }
    
    // 3. SPATIAL PATH (QNN) - First convolution
    quantized_conv1d_rv32x(buffers->input_buf, conv1_weight, conv1_weight_scale,
                          buffers->conv1_out,
                          N_FEATURES, CONV1_FILTERS, INPUT_SIZE, 5, 2,
                          buffers->precision_mode);
    
    // Add bias and ReLU
    for (int i = 0; i < CONV1_FILTERS * CONV1_OUT_SIZE; i++) {
        int channel = i / CONV1_OUT_SIZE;
        int32_t bias_term = ((int32_t)conv1_bias[channel] * conv1_bias_scale) >> FIXED_SCALE;
        int32_t biased = buffers->conv1_out[i] + bias_term;
        buffers->conv1_out[i] = biased > 0 ? biased : 0;
    }
    
    // 4. SPATIAL PATH - Second convolution
    quantized_conv1d_rv32x((int8_t*)buffers->conv1_out, conv2_weight, conv2_weight_scale,
                          buffers->conv2_out,
                          CONV1_FILTERS, CONV2_FILTERS, CONV1_OUT_SIZE, 3, 2,
                          buffers->precision_mode);
    
    // Add bias and ReLU
    for (int i = 0; i < CONV2_FILTERS * CONV2_OUT_SIZE; i++) {
        int channel = i / CONV2_OUT_SIZE;
        int32_t bias_term = ((int32_t)conv2_bias[channel] * conv2_bias_scale) >> FIXED_SCALE;
        int32_t biased = buffers->conv2_out[i] + bias_term;
        buffers->conv2_out[i] = biased > 0 ? biased : 0;
    }
    
    // 5. Prepare LIF neuron input
    for (int t = 0; t < TIME_STEPS; t++) {
        for (int i = 0; i < SNN_INPUT_SIZE; i++) {
            int spatial_idx = t * SNN_INPUT_SIZE + i;
            if (spatial_idx < (CONV2_FILTERS * CONV2_OUT_SIZE)) {
                buffers->snn_input[t * SNN_INPUT_SIZE + i] = 
                    (int8_t)(buffers->conv2_out[spatial_idx] >> FIXED_SCALE);
            } else {
                buffers->snn_input[t * SNN_INPUT_SIZE + i] = 0;
            }
        }
    }
    
    // 6. LIF neurons for spike generation
    for (int t = 0; t < TIME_STEPS; t++) {
        for (int i = 0; i < SNN_HIDDEN; i++) {
            int8_t input_val = buffers->snn_input[t * SNN_INPUT_SIZE + (i % SNN_INPUT_SIZE)];
            int8_t membrane_potential = (t == 0) ? 0 : buffers->spike_train[(t-1) * SNN_HIDDEN + i];
            
            buffers->spike_train[t * SNN_HIDDEN + i] = 
                custom2_lif(input_val, membrane_potential, 50);
        }
    }
    
    // 7. LSTM layer for temporal dynamics
    quantized_lstm_layer_rv32x(buffers->spike_train, buffers->lstm_state,
                              lstm_weight_ih, lstm_weight_hh, lstm_bias_ih,
                              lstm_weight_ih_scale, lstm_weight_hh_scale, lstm_bias_ih_scale,
                              buffers->lstm_state, TIME_STEPS, LSTM_HIDDEN,
                              buffers->precision_mode);
    
    // 8. Feature router
    int8_t spatial_features[CONV2_FILTERS];
    for (int i = 0; i < CONV2_FILTERS; i++) {
        int32_t sum = 0;
        for (int j = 0; j < CONV2_OUT_SIZE; j++) {
            sum += buffers->conv2_out[i * CONV2_OUT_SIZE + j];
        }
        spatial_features[i] = (int8_t)((sum / CONV2_OUT_SIZE) >> FIXED_SCALE);
    }
    
    uint8_t attention_weight = feature_router(spatial_features, buffers->lstm_state);
    buffers->router_output[0] = attention_weight;
    buffers->router_output[1] = 127 - attention_weight;
    
    // 9. Dynamic fusion
    int8_t snn_features[LSTM_HIDDEN];
    int8_t qnn_features[CONV2_FILTERS];
    
    for (int i = 0; i < LSTM_HIDDEN; i++) {
        snn_features[i] = buffers->lstm_state[i];
    }
    for (int i = 0; i < CONV2_FILTERS; i++) {
        qnn_features[i] = spatial_features[i];
    }
    
    dynamic_fusion_rv32x(snn_features, qnn_features, attention_weight,
                        buffers->fused_features, FUSION_SIZE,
                        buffers->precision_mode);
    
    // 10. Classification layers
    int min_size = (FC1_SIZE < FUSION_SIZE) ? FC1_SIZE : FUSION_SIZE;
    
    for (int i = 0; i < min_size; i++) {
        int32_t sum = 0;
        for (int j = 0; j < FUSION_SIZE; j++) {
            int8_t fused_val = (int8_t)(buffers->fused_features[j] >> FIXED_SCALE);
            int weight_idx = i * FUSION_SIZE + j;
            sum = custom1_mac(fused_val, fc1_weight[weight_idx], sum);
        }
        buffers->fc1_out[i] = ((int64_t)sum * fc1_weight_scale) >> FIXED_SCALE;
        buffers->fc1_out[i] += ((int32_t)fc1_bias[i] * fc1_bias_scale) >> FIXED_SCALE;
        if (buffers->fc1_out[i] < 0) buffers->fc1_out[i] = 0;
    }
    
    // Output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int32_t sum = 0;
        for (int j = 0; j < FC1_SIZE; j++) {
            int8_t fc1_val = (int8_t)(buffers->fc1_out[j] >> FIXED_SCALE);
            int weight_idx = i * FC1_SIZE + j;
            sum = custom1_mac(fc1_val, fc2_weight[weight_idx], sum);
        }
        buffers->output[i] = ((int64_t)sum * fc2_weight_scale) >> FIXED_SCALE;
        buffers->output[i] += ((int32_t)fc2_bias[i] * fc2_bias_scale) >> FIXED_SCALE;
    }
    
    // End cycle counting
    #ifdef ENABLE_BENCHMARKING
    cycle_t end_cycles = get_cycle_count();
    uint32_t cycles = (uint32_t)(end_cycles - start_cycles);
    buffers->total_cycles = cycles;
    buffers->inference_count++;
    #endif
}
