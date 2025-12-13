#include "ops/math_ops.h"
#include "firmware/model/model_weights.h"
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
// Enlever: #include <string.h>

// ==================== QUANTIZED OPERATIONS WITH RV32X-SQ ====================

void quantized_conv1d_rv32x(const int8_t *input, const int8_t *kernel,
                           int32_t kernel_scale, int32_t *output,
                           int in_channels, int out_channels,
                           int length, int kernel_size, int stride,
                           uint8_t precision_mode) {
    const int out_length = length / stride;
    
    for (int oc = 0; oc < out_channels; oc++) {
        for (int i = 0; i < out_length; i++) {
            int32_t sum = 0;
            const int in_pos = i * stride;
            const int8_t *k_ptr = &kernel[oc * in_channels * kernel_size];
            
            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int input_idx = (ic * length) + in_pos + k;
                    int kernel_idx = (ic * kernel_size) + k;
                    
                    // Use custom1_mac for 4-bit MAC when in medium/low precision mode
                    if (precision_mode >= PRECISION_MEDIUM) {
                        sum = custom1_mac(input[input_idx], k_ptr[kernel_idx], sum);
                    } else {
                        sum += input[input_idx] * k_ptr[kernel_idx];
                    }
                }
            }
            
            // Apply scaling based on precision mode
            int32_t scale_factor = kernel_scale;
            if (precision_mode == PRECISION_LOW) {
                scale_factor = scale_factor >> 1;
            }
            
            output[oc * out_length + i] = ((int64_t)sum * scale_factor) >> FIXED_SCALE;
        }
    }
}

void quantized_lstm_layer_rv32x(const int8_t *input, int8_t *hidden_state,
                               const int8_t *w_ih, const int8_t *w_hh, const int8_t *bias,
                               int32_t w_ih_scale, int32_t w_hh_scale, int32_t bias_scale,
                               int8_t *output, int seq_len, int hidden_size,
                               uint8_t precision_mode) {
    // Créer un tableau temporaire (allocation sur la pile)
    int8_t tmp_state[hidden_size];
    const int32_t combined_scale = ((int64_t)w_ih_scale * w_hh_scale) >> FIXED_SCALE;
    
    for (int t = 0; t < seq_len; t++) {
        const int8_t *current_input = &input[t * hidden_size];
        
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum_ih = 0;
            int32_t sum_hh = 0;
            
            for (int j = 0; j < hidden_size; j++) {
                // Use custom1_mac for efficient MAC
                if (precision_mode >= PRECISION_MEDIUM) {
                    sum_ih = custom1_mac(current_input[j], w_ih[i * hidden_size + j], sum_ih);
                    sum_hh = custom1_mac(hidden_state[j], w_hh[i * hidden_size + j], sum_hh);
                } else {
                    sum_ih += current_input[j] * w_ih[i * hidden_size + j];
                    sum_hh += hidden_state[j] * w_hh[i * hidden_size + j];
                }
            }
            
            int32_t sum = ((int64_t)(sum_ih + sum_hh) * combined_scale) >> FIXED_SCALE;
            sum += (bias[i] * bias_scale) >> FIXED_SCALE;
            
            // Apply LIF neuron dynamics using custom2_lif
            if (precision_mode >= PRECISION_MEDIUM) {
                tmp_state[i] = custom2_lif((int8_t)(sum >> FIXED_SCALE), 
                                         hidden_state[i], 50);
            } else {
                // Simple activation for high precision
                tmp_state[i] = (int8_t)(sum >> FIXED_SCALE);
            }
        }
        
        // Update hidden state
        for (int i = 0; i < hidden_size; i++) {
            hidden_state[i] = tmp_state[i];
        }
    }
    
    // Copy final state to output
    for (int i = 0; i < hidden_size; i++) {
        output[i] = hidden_state[i];
    }
}

void dynamic_fusion_rv32x(const int8_t *snn_features, const int8_t *qnn_features,
                         uint8_t attention_weight, int32_t *fused_output,
                         int fusion_size, uint8_t precision_mode) {
    // α_t·SNN + (1-α_t)·QNN using custom3_fusion
    
    for (int i = 0; i < fusion_size; i++) {
        int8_t snn_val = snn_features[i];
        int8_t qnn_val = qnn_features[i];
        
        if (precision_mode >= PRECISION_MEDIUM) {
            // Use custom3_fusion for efficient fusion
            fused_output[i] = custom3_fusion(snn_val, qnn_val, attention_weight);
        } else {
            // High precision fusion
            int16_t snn_contrib = (snn_val * attention_weight);
            int16_t qnn_contrib = (qnn_val * (127 - attention_weight));
            fused_output[i] = (snn_contrib + qnn_contrib) >> 7;
        }
    }
}

// ==================== HELPER FUNCTIONS ====================

int8_t clamp_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

int32_t tanh_approx(int32_t x) {
    // Simple approximation suitable for edge devices
    if (x < -4 * FIXED_SCALE_VAL) return -FIXED_SCALE_VAL;
    if (x > 4 * FIXED_SCALE_VAL) return FIXED_SCALE_VAL;
    
    // x - x³/3 approximation for small values
    int64_t x_sq = ((int64_t)x * x) >> FIXED_SCALE;
    int64_t x_cu = ((int64_t)x_sq * x) >> FIXED_SCALE;
    return x - (x_cu / 3);
}

int32_t sigmoid_approx(int32_t x) {
    // Fixed-point sigmoid approximation
    if (x < -8 * FIXED_SCALE_VAL) return 0;
    if (x > 8 * FIXED_SCALE_VAL) return FIXED_SCALE_VAL;
    
    // Piecewise linear approximation
    if (x < 0) {
        return (x + 8 * FIXED_SCALE_VAL) >> 4;
    } else {
        return FIXED_SCALE_VAL - ((8 * FIXED_SCALE_VAL - x) >> 4);
    }
}

// ==================== THERMAL MANAGEMENT ====================
// Cette fonction doit être implémentée ici car elle n'est pas dans model_weights.c

void thermal_management(HybridModelBuffers* buffers) {
    static uint32_t thermal_counter = 0;
    
    // Simple temperature simulation
    if (buffers->temperature < 0) {
        buffers->temperature = 25 + (thermal_counter / 10);
    }
    
    // Adjust precision based on temperature
    if (buffers->temperature > TEMP_THRESHOLD_HIGH) {
        buffers->precision_mode = PRECISION_LOW;
    } else if (buffers->temperature > TEMP_THRESHOLD_MEDIUM) {
        buffers->precision_mode = PRECISION_MEDIUM;
    } else {
        buffers->precision_mode = PRECISION_HIGH;
    }
    
    thermal_counter++;
    
    if (thermal_counter > 100 && buffers->temperature > TEMP_THRESHOLD_MEDIUM) {
        thermal_counter = 0;
    }
}

// ==================== FEATURE ROUTER ====================

uint8_t feature_router(const int8_t* spatial_features, const int8_t* temporal_features) {
    // Simplified routing: use temporal signal energy as decision metric
    int32_t temporal_energy = 0;
    for (int i = 0; i < LSTM_HIDDEN; i++) {
        temporal_energy += temporal_features[i] * temporal_features[i];
    }
    
    int32_t spatial_energy = 0;
    for (int i = 0; i < CONV2_FILTERS; i++) {
        spatial_energy += spatial_features[i] * spatial_features[i];
    }
    
    // Dynamic attention weight based on energy ratio
    if (temporal_energy > spatial_energy * 2) {
        return 89;  // α=0.7
    } else if (spatial_energy > temporal_energy * 2) {
        return 51;  // α=0.4
    } else {
        return 70;  // α≈0.55
    }
}

// ==================== RV32X-SQ CUSTOM INSTRUCTIONS ====================
// Ces fonctions doivent être implémentées ici

int32_t custom1_mac(int8_t a, int8_t b, int32_t acc) {
    // Simulated 4-bit MAC operation
    int16_t product = (int16_t)a * (int16_t)b;
    return acc + product;
}

int8_t custom2_lif(int8_t input, int8_t membrane_potential, int8_t threshold) {
    // Simulated LIF neuron update
    int16_t new_potential = membrane_potential + input;
    
    if (new_potential >= threshold) {
        // Spike generation
        return 127;  // Spike
    }
    
    // Leak
    new_potential = (new_potential * 95) / 100;
    
    // Clip to int8 range
    if (new_potential > 127) return 127;
    if (new_potential < -128) return -128;
    return (int8_t)new_potential;
}

int8_t custom3_fusion(int8_t snn_out, int8_t qnn_out, int8_t attention_weight) {
    // Simulated attention-based fusion
    int16_t snn_weighted = snn_out * attention_weight;
    int16_t qnn_weighted = qnn_out * (127 - attention_weight);
    
    int16_t fused = (snn_weighted + qnn_weighted) / 127;
    
    if (fused > 127) return 127;
    if (fused < -128) return -128;
    return (int8_t)fused;
}
