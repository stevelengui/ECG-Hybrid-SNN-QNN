#include "ops/math_ops.h"
#include "firmware/model/model_weights.h"
#include "thermal_manager.h"
#include <stdint.h>
#include <stddef.h>

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
                    if (precision_mode >= THERMAL_PRECISION_MEDIUM) {
                        sum = custom1_mac(input[input_idx], k_ptr[kernel_idx], sum);
                    } else {
                        sum += input[input_idx] * k_ptr[kernel_idx];
                    }
                }
            }
            
            // Apply scaling based on precision mode
            int32_t scale_factor = kernel_scale;
            if (precision_mode == THERMAL_PRECISION_LOW) {
                scale_factor = scale_factor >> 1;  // Réduction pour 2-bit
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
    // Tableau temporaire sur la pile
    int8_t tmp_state[hidden_size];
    const int32_t combined_scale = ((int64_t)w_ih_scale * w_hh_scale) >> FIXED_SCALE;
    
    for (int t = 0; t < seq_len; t++) {
        const int8_t *current_input = &input[t * hidden_size];
        
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum_ih = 0;
            int32_t sum_hh = 0;
            
            for (int j = 0; j < hidden_size; j++) {
                // Utiliser custom1_mac pour MAC efficace
                if (precision_mode >= THERMAL_PRECISION_MEDIUM) {
                    sum_ih = custom1_mac(current_input[j], w_ih[i * hidden_size + j], sum_ih);
                    sum_hh = custom1_mac(hidden_state[j], w_hh[i * hidden_size + j], sum_hh);
                } else {
                    sum_ih += current_input[j] * w_ih[i * hidden_size + j];
                    sum_hh += hidden_state[j] * w_hh[i * hidden_size + j];
                }
            }
            
            int32_t sum = ((int64_t)(sum_ih + sum_hh) * combined_scale) >> FIXED_SCALE;
            sum += (bias[i] * bias_scale) >> FIXED_SCALE;
            
            // Appliquer dynamique LIF neuron avec custom2_lif
            if (precision_mode >= THERMAL_PRECISION_MEDIUM) {
                tmp_state[i] = custom2_lif((int8_t)(sum >> FIXED_SCALE), 
                                         hidden_state[i], 50);
            } else {
                // Activation simple pour haute précision
                tmp_state[i] = (int8_t)(sum >> FIXED_SCALE);
            }
        }
        
        // Mettre à jour état caché
        for (int i = 0; i < hidden_size; i++) {
            hidden_state[i] = tmp_state[i];
        }
    }
    
    // Copier état final vers sortie
    for (int i = 0; i < hidden_size; i++) {
        output[i] = hidden_state[i];
    }
}

void dynamic_fusion_rv32x(const int8_t *snn_features, const int8_t *qnn_features,
                         uint8_t attention_weight, int32_t *fused_output,
                         int fusion_size, uint8_t precision_mode) {
    // α_t·SNN + (1-α_t)·QNN avec custom3_fusion
    
    for (int i = 0; i < fusion_size; i++) {
        int8_t snn_val = snn_features[i];
        int8_t qnn_val = qnn_features[i];
        
        if (precision_mode >= THERMAL_PRECISION_MEDIUM) {
            // Utiliser custom3_fusion pour fusion efficace
            fused_output[i] = custom3_fusion(snn_val, qnn_val, attention_weight);
        } else {
            // Fusion haute précision
            int16_t snn_contrib = (snn_val * attention_weight);
            int16_t qnn_contrib = (qnn_val * (127 - attention_weight));
            fused_output[i] = (snn_contrib + qnn_contrib) >> 7; // Division par 127
        }
    }
}

// ==================== THERMAL MANAGEMENT UPDATED ====================

void thermal_management(HybridModelBuffers* buffers) {
    static ThermalManager tm;
    static uint8_t initialized = 0;
    
    // Initialiser au premier appel
    if (!initialized) {
        thermal_application_domain_t domain;
        
        // Détecter domaine basé sur configuration compilation
#ifdef ECG_MODEL
        domain = THERMAL_DOMAIN_MEDICAL;
#elif defined(VIBRATION_MODEL)
        domain = THERMAL_DOMAIN_INDUSTRIAL;
#else
        domain = THERMAL_DOMAIN_INDUSTRIAL;  // Défaut conservateur
#endif
        
        thermal_manager_init(&tm, domain);
        initialized = 1;
    }
    
    // SIMULATION THERMIQUE (dans vrai système, lire capteurs)
    // --------------------------------------------------------
    static uint16_t sim_chip_temp = 250;      // 25.0°C initial
    static uint16_t sim_ambient_temp = 250;   // 25.0°C initial
    static uint32_t thermal_counter = 0;
    
    thermal_counter++;
    
    // Simulation d'échauffement pendant l'inférence
    if (thermal_counter % 3 == 0) {
        // Chaque inférence ajoute de la chaleur
        sim_chip_temp += 5;  // +0.5°C
        
        // Refroidissement naturel
        if (sim_chip_temp > sim_ambient_temp + 10) {
            sim_chip_temp -= 2;  // -0.2°C
        }
        
        // Limiter température max
        if (sim_chip_temp > 850) {  // 85.0°C max
            sim_chip_temp = 600;    // Retour à 60.0°C (surchauffe évitée)
        }
    }
    
    // Variation ambiante périodique
    if (thermal_counter % 100 == 0) {
        // Simuler environnement variable
        static uint16_t ambient_pattern[] = {250, 280, 300, 320, 350, 400, 350, 320, 300, 280};
        static uint8_t pattern_index = 0;
        
        sim_ambient_temp = ambient_pattern[pattern_index];
        pattern_index = (pattern_index + 1) % (sizeof(ambient_pattern)/sizeof(ambient_pattern[0]));
    }
    
    // Mettre à jour le gestionnaire thermique
    thermal_manager_update(&tm, sim_chip_temp, sim_ambient_temp);
    
    // Appliquer le mode de précision déterminé
    buffers->precision_mode = (uint8_t)thermal_manager_get_mode(&tm);
    
    // Stocker ΔT pour monitoring (convertir en °C)
    int16_t delta_t = thermal_manager_get_delta_t(&tm);
    buffers->temperature = (int16_t)(delta_t / 10);
    
    // Log thermique si activé
#ifdef ENABLE_THERMAL_LOGGING
    static uint32_t log_counter = 0;
    if (log_counter++ % 50 == 0) {
        // Dans vrai système: envoyer via UART
        // printf("Thermal: Chip=%d.%d°C, Ambient=%d.%d°C, ΔT=%d.%d°C, Mode=%d\n",
        //        sim_chip_temp/10, sim_chip_temp%10,
        //        sim_ambient_temp/10, sim_ambient_temp%10,
        //        delta_t/10, delta_t%10,
        //        buffers->precision_mode);
    }
#endif
    
    // Gestion DVFS si activée
#ifdef ENABLE_DVFS
    // Ajuster fréquence/voltage basé sur marge thermique
    uint8_t thermal_margin = calculate_thermal_margin(&tm);
    
    if (thermal_margin < 20) {  // Marge < 20%
        // Réduction agressive
        // set_cpu_frequency(CPU_FREQ * 0.6);
        // set_core_voltage(CORE_VOLT * 0.85);
    } else if (thermal_margin < 50) {  // Marge < 50%
        // Réduction modérée
        // set_cpu_frequency(CPU_FREQ * 0.8);
        // set_core_voltage(CORE_VOLT * 0.9);
    } else {
        // Fréquence nominale
        // set_cpu_frequency(CPU_FREQ);
        // set_core_voltage(CORE_VOLT);
    }
#endif
}

// ==================== FEATURE ROUTER ====================

uint8_t feature_router(const int8_t* spatial_features, const int8_t* temporal_features) {
    // Routage simplifié: utiliser l'énergie temporelle comme métrique de décision
    int32_t temporal_energy = 0;
    for (int i = 0; i < LSTM_HIDDEN; i++) {
        temporal_energy += temporal_features[i] * temporal_features[i];
    }
    
    int32_t spatial_energy = 0;
    for (int i = 0; i < CONV2_FILTERS; i++) {
        spatial_energy += spatial_features[i] * spatial_features[i];
    }
    
    // Poids d'attention dynamique basé sur ratio d'énergie
    if (temporal_energy > spatial_energy * 2) {
        return 89;  // α=0.7 (favorise SNN)
    } else if (spatial_energy > temporal_energy * 2) {
        return 51;  // α=0.4 (favorise QNN)
    } else {
        return 70;  // α≈0.55 (équilibre)
    }
}

// ==================== HELPER FUNCTIONS ====================

int8_t clamp_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

int32_t tanh_approx(int32_t x) {
    // Approximation simple pour dispositifs edge
    if (x < -4 * FIXED_SCALE_VAL) return -FIXED_SCALE_VAL;
    if (x > 4 * FIXED_SCALE_VAL) return FIXED_SCALE_VAL;
    
    // Approximation x - x³/3 pour petites valeurs
    int64_t x_sq = ((int64_t)x * x) >> FIXED_SCALE;
    int64_t x_cu = ((int64_t)x_sq * x) >> FIXED_SCALE;
    return x - (x_cu / 3);
}

int32_t sigmoid_approx(int32_t x) {
    // Approximation sigmoïde en fixed-point
    if (x < -8 * FIXED_SCALE_VAL) return 0;
    if (x > 8 * FIXED_SCALE_VAL) return FIXED_SCALE_VAL;
    
    // Approximation linéaire par morceaux
    if (x < 0) {
        return (x + 8 * FIXED_SCALE_VAL) >> 4;
    } else {
        return FIXED_SCALE_VAL - ((8 * FIXED_SCALE_VAL - x) >> 4);
    }
}

// ==================== RV32X-SQ CUSTOM INSTRUCTIONS ====================

int32_t custom1_mac(int8_t a, int8_t b, int32_t acc) {
    // Simulation d'opération MAC 4-bit
    int16_t product = (int16_t)a * (int16_t)b;
    return acc + product;
}

int8_t custom2_lif(int8_t input, int8_t membrane_potential, int8_t threshold) {
    // Simulation de mise à jour de neurone LIF
    int16_t new_potential = membrane_potential + input;
    
    if (new_potential >= threshold) {
        // Génération de spike
        return 127;  // Spike
    }
    
    // Fuite
    new_potential = (new_potential * 95) / 100;
    
    // Limiter à la plage int8
    if (new_potential > 127) return 127;
    if (new_potential < -128) return -128;
    return (int8_t)new_potential;
}

int8_t custom3_fusion(int8_t snn_out, int8_t qnn_out, int8_t attention_weight) {
    // Simulation de fusion basée sur attention
    int16_t snn_weighted = snn_out * attention_weight;
    int16_t qnn_weighted = qnn_out * (127 - attention_weight);
    
    int16_t fused = (snn_weighted + qnn_weighted) / 127;
    
    if (fused > 127) return 127;
    if (fused < -128) return -128;
    return (int8_t)fused;
}
