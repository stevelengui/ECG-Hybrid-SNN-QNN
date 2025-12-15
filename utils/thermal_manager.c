#include "thermal_manager.h"

// Fonction absolue simple (sans utiliser abs() de stdlib)
static int16_t abs_int16(int16_t x) {
    return (x < 0) ? -x : x;
}

void thermal_manager_init(ThermalManager* tm, thermal_application_domain_t domain) {
    tm->current_mode = THERMAL_PRECISION_HIGH;
    tm->domain = domain;
    tm->inference_count = 0;
    tm->chip_temperature = 250;     // 25.0°C
    tm->ambient_temperature = 250;  // 25.0°C
    tm->delta_t = 0;
    tm->thermal_violation = 0;
    tm->thermal_warning = 0;
    tm->thermal_integral = 0;
    tm->last_action_temp = 0;
    tm->last_action_delta_t = 0;
}

void thermal_manager_update(ThermalManager* tm, int16_t new_chip_temp, int16_t new_ambient_temp) {
    // Mettre à jour les températures
    tm->chip_temperature = new_chip_temp;
    tm->ambient_temperature = new_ambient_temp;
    
    // Calculer ΔT (en dixièmes de °C)
    tm->delta_t = new_chip_temp - new_ambient_temp;
    
    // Mettre à jour l'intégral thermique
    if (tm->delta_t > 0) {
        tm->thermal_integral += tm->delta_t;
    }
    
    // Déterminer la limite ΔT selon le domaine
    int16_t delta_t_max = (tm->domain == THERMAL_DOMAIN_MEDICAL) ? 
                         DELTA_T_MEDICAL_MAX : DELTA_T_INDUSTRIAL_MAX;
    
    int16_t delta_t_warning = (delta_t_max * 8) / 10;  // 80% de la limite
    
    // Vérifier les violations et alertes
    tm->thermal_violation = (tm->delta_t > delta_t_max) ? 1 : 0;
    tm->thermal_warning = (tm->delta_t > delta_t_warning) ? 1 : 0;
    
    // DÉCISION HIÉRARCHIQUE THERMIQUE
    thermal_precision_mode_t new_mode = THERMAL_PRECISION_HIGH;
    
    // NIVEAU 1: Violation ΔT (la plus stricte)
    if (tm->thermal_violation) {
        new_mode = THERMAL_PRECISION_LOW;  // Force 2-bit
    }
    // NIVEAU 2: Alerte ΔT
    else if (tm->thermal_warning) {
        if (tm->current_mode == THERMAL_PRECISION_HIGH) {
            new_mode = THERMAL_PRECISION_MEDIUM;  // 8→4 bits
        } else if (tm->current_mode == THERMAL_PRECISION_MEDIUM) {
            new_mode = THERMAL_PRECISION_LOW;     // 4→2 bits
        }
    }
    // NIVEAU 3: Température absolue
    else if (new_chip_temp >= TEMP_ABSOLUTE_CRITICAL) {
        new_mode = THERMAL_PRECISION_LOW;  // Urgence
    } else if (new_chip_temp >= TEMP_ABSOLUTE_HIGH) {
        new_mode = THERMAL_PRECISION_LOW;  // 4→2 bits
    } else if (new_chip_temp >= TEMP_ABSOLUTE_MEDIUM) {
        if (tm->current_mode == THERMAL_PRECISION_HIGH) {
            new_mode = THERMAL_PRECISION_MEDIUM;  // 8→4 bits
        }
    }
    
    // Appliquer l'hystérésis (utiliser notre fonction abs_int16)
    if (tm->current_mode != new_mode) {
        // Vérifier si le changement est significatif
        int16_t temp_diff = new_chip_temp - tm->last_action_temp;
        int16_t delta_diff = tm->delta_t - tm->last_action_delta_t;
        
        // Utiliser abs_int16 au lieu de abs()
        if (abs_int16(temp_diff) >= 20 || abs_int16(delta_diff) >= 10) {  // 2.0°C et 1.0°C
            tm->current_mode = new_mode;
            tm->last_action_temp = new_chip_temp;
            tm->last_action_delta_t = tm->delta_t;
        }
    }
    
    tm->inference_count++;
}

thermal_precision_mode_t thermal_manager_get_mode(ThermalManager* tm) {
    return tm->current_mode;
}

int16_t thermal_manager_get_delta_t(ThermalManager* tm) {
    return tm->delta_t;
}

uint8_t thermal_manager_is_violating(ThermalManager* tm) {
    return tm->thermal_violation;
}

uint8_t thermal_manager_is_warning(ThermalManager* tm) {
    return tm->thermal_warning;
}

void thermal_manager_set_ambient(ThermalManager* tm, int16_t ambient_temp) {
    tm->ambient_temperature = ambient_temp;
}

int16_t calculate_safe_chip_temp(int16_t ambient_temp, thermal_application_domain_t domain) {
    int16_t delta_max = (domain == THERMAL_DOMAIN_MEDICAL) ? 
                       DELTA_T_MEDICAL_MAX : DELTA_T_INDUSTRIAL_MAX;
    return ambient_temp + delta_max;
}

uint8_t calculate_thermal_margin(ThermalManager* tm) {
    int16_t delta_max = (tm->domain == THERMAL_DOMAIN_MEDICAL) ? 
                       DELTA_T_MEDICAL_MAX : DELTA_T_INDUSTRIAL_MAX;
    
    if (tm->delta_t >= delta_max) return 0;
    
    // Calculer la marge en pourcentage (0-100)
    int32_t margin = (delta_max - tm->delta_t) * 100;
    margin = margin / delta_max;
    
    return (margin > 100) ? 100 : (margin < 0) ? 0 : (uint8_t)margin;
}
