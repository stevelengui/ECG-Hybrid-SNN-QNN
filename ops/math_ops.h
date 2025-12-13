#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <stdint.h>
#include <stddef.h>

// NE PAS redéclarer les fonctions qui sont déjà dans model_weights.h
// Les prototypes suivants sont déjà définis dans model_weights.h:
// - thermal_management()
// - feature_router()
// - custom1_mac()
// - custom2_lif()
// - custom3_fusion()

// Fonctions de convolution optimisées (UNIQUEMENT celles qui ne sont pas dans model_weights.h)
void quantized_conv1d_rv32x(const int8_t* input, const int8_t* weight, int32_t weight_scale,
                           int32_t* output, int in_channels, int out_channels,
                           int input_size, int kernel_size, int stride,
                           uint8_t precision_mode);

void quantized_lstm_layer_rv32x(const int8_t* input, int8_t* hidden_state,
                               const int8_t* weight_ih, const int8_t* weight_hh,
                               const int8_t* bias_ih, int32_t weight_ih_scale,
                               int32_t weight_hh_scale, int32_t bias_ih_scale,
                               int8_t* output, int seq_len, int hidden_size,
                               uint8_t precision_mode);

void dynamic_fusion_rv32x(const int8_t* snn_features, const int8_t* qnn_features,
                         uint8_t attention_weight, int32_t* output, int output_size,
                         uint8_t precision_mode);

// Fonctions utilitaires (UNIQUEMENT celles qui ne sont pas dans model_weights.h)
int8_t clamp_int8(int32_t x);
int32_t tanh_approx(int32_t x);
int32_t sigmoid_approx(int32_t x);

#endif // MATH_OPS_H
