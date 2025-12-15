#ifndef THERMAL_MANAGER_H
#define THERMAL_MANAGER_H

#include <stdint.h>

// Définitions des domaines avec valeurs entières
typedef enum {
    THERMAL_DOMAIN_MEDICAL = 0,     // ECG
    THERMAL_DOMAIN_INDUSTRIAL = 1,  // Vibration
    THERMAL_DOMAIN_AUDIO = 2        // Future
} thermal_application_domain_t;

// Modes de précision
typedef enum {
    THERMAL_PRECISION_HIGH = 0,    // 8-bit
    THERMAL_PRECISION_MEDIUM = 1,  // 4-bit  
    THERMAL_PRECISION_LOW = 2      // 2-bit
} thermal_precision_mode_t;

// Seuils thermiques (en dixièmes de °C pour éviter les floats)
#define TEMP_ABSOLUTE_MEDIUM    500  // 50.0°C
#define TEMP_ABSOLUTE_HIGH      700  // 70.0°C
#define TEMP_ABSOLUTE_CRITICAL  850  // 85.0°C

// Contraintes ΔT (dixièmes de °C)
#define DELTA_T_MEDICAL_MAX     50   // 5.0°C
#define DELTA_T_INDUSTRIAL_MAX  70   // 7.0°C

// Structure sans floats
typedef struct {
    thermal_precision_mode_t current_mode;
    thermal_application_domain_t domain;
    uint32_t inference_count;
    int16_t chip_temperature;      // Température puce (dixièmes de °C)
    int16_t ambient_temperature;   // Température ambiante (dixièmes de °C)
    int16_t delta_t;               // ΔT calculé (dixièmes de °C)
    uint8_t thermal_violation;     // Violation ΔT
    uint8_t thermal_warning;       // Alerte préventive
    uint32_t thermal_integral;     // ∫ΔT dt
    int16_t last_action_temp;      // Pour hystérésis
    int16_t last_action_delta_t;   // Pour hystérésis ΔT
} ThermalManager;

// Prototypes simplifiés sans floats
void thermal_manager_init(ThermalManager* tm, thermal_application_domain_t domain);
void thermal_manager_update(ThermalManager* tm, int16_t new_chip_temp, int16_t new_ambient_temp);
thermal_precision_mode_t thermal_manager_get_mode(ThermalManager* tm);
void thermal_manager_adjust_precision(ThermalManager* tm, uint32_t predicted_cycles);
int16_t thermal_manager_get_delta_t(ThermalManager* tm);
uint8_t thermal_manager_is_violating(ThermalManager* tm);
uint8_t thermal_manager_is_warning(ThermalManager* tm);
void thermal_manager_set_ambient(ThermalManager* tm, int16_t ambient_temp);
int16_t calculate_safe_chip_temp(int16_t ambient_temp, thermal_application_domain_t domain);
uint8_t calculate_thermal_margin(ThermalManager* tm);  // Pourcentage 0-100
thermal_application_domain_t get_current_thermal_domain(void);

#endif // THERMAL_MANAGER_H
