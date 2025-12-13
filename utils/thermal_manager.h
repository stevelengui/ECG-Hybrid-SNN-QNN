#ifndef THERMAL_MANAGER_H
#define THERMAL_MANAGER_H

#include <stdint.h>

// Precision modes
typedef enum {
    PRECISION_HIGH = 0,    // 8-bit mode
    PRECISION_MEDIUM = 1,  // 4-bit mode
    PRECISION_LOW = 2      // 2-bit mode
} precision_mode_t;

// Thermal manager state
typedef struct {
    precision_mode_t current_mode;
    uint32_t inference_count;
    int16_t temperature;
    uint8_t thermal_throttle;
} ThermalManager;

// Function prototypes
void thermal_manager_init(ThermalManager* tm);
void thermal_manager_update(ThermalManager* tm, int16_t new_temp);
precision_mode_t thermal_manager_get_mode(ThermalManager* tm);
void thermal_manager_adjust_precision(ThermalManager* tm, uint32_t predicted_cycles);

#endif // THERMAL_MANAGER_H
