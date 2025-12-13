#include "model_weights.h"

// ==================== WEIGHT DEFINITIONS ====================
const int8_t conv1_weight[] = {
    #include "arrays/conv1_weight_array.txt"
};
const int32_t conv1_weight_scale = 1;

const int8_t conv1_bias[] = {
    #include "arrays/conv1_bias_array.txt"
};
const int32_t conv1_bias_scale = 1;

const int8_t conv2_weight[] = {
    #include "arrays/conv2_weight_array.txt"
};
const int32_t conv2_weight_scale = 2;

const int8_t conv2_bias[] = {
    #include "arrays/conv2_bias_array.txt"
};
const int32_t conv2_bias_scale = 1;

const int8_t lstm_weight_ih[] = {
    #include "arrays/lstm_weight_ih_array.txt"
};
const int32_t lstm_weight_ih_scale = 2;

const int8_t lstm_weight_hh[] = {
    #include "arrays/lstm_weight_hh_array.txt"
};
const int32_t lstm_weight_hh_scale = 3;

const int8_t lstm_bias_ih[] = {
    #include "arrays/lstm_bias_ih_array.txt"
};
const int32_t lstm_bias_ih_scale = 1;

const int8_t lstm_bias_hh[] = {
    #include "arrays/lstm_bias_hh_array.txt"
};
const int32_t lstm_bias_hh_scale = 1;

const int8_t router_fc1_weight[] = {
    #include "arrays/router_fc1_weight_array.txt"
};
const int32_t router_fc1_weight_scale = 1;

const int8_t router_fc1_bias[] = {
    #include "arrays/router_fc1_bias_array.txt"
};
const int32_t router_fc1_bias_scale = 0;

const int8_t router_fc2_weight[] = {
    #include "arrays/router_fc2_weight_array.txt"
};
const int32_t router_fc2_weight_scale = 1;

const int8_t router_fc2_bias[] = {
    #include "arrays/router_fc2_bias_array.txt"
};
const int32_t router_fc2_bias_scale = 0;

const int8_t fusion_weight[] = {
    #include "arrays/fusion_weight_array.txt"
};
const int32_t fusion_weight_scale = 3;

const int8_t fusion_bias[] = {
    #include "arrays/fusion_bias_array.txt"
};
const int32_t fusion_bias_scale = 1;

const int8_t fc1_weight[] = {
    #include "arrays/fc1_weight_array.txt"
};
const int32_t fc1_weight_scale = 2;

const int8_t fc1_bias[] = {
    #include "arrays/fc1_bias_array.txt"
};
const int32_t fc1_bias_scale = 0;

const int8_t fc2_weight[] = {
    #include "arrays/fc2_weight_array.txt"
};
const int32_t fc2_weight_scale = 3;

const int8_t fc2_bias[] = {
    #include "arrays/fc2_bias_array.txt"
};
const int32_t fc2_bias_scale = 2;
