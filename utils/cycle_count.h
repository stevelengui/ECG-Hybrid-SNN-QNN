#ifndef CYCLE_COUNT_H
#define CYCLE_COUNT_H

#include <stdint.h>

// Type pour stocker les cycles
#if defined(__riscv_xlen) && __riscv_xlen == 64
    typedef uint64_t cycle_t;
#else
    typedef uint32_t cycle_t;
#endif

// Initialisation
void cycle_count_init(uint32_t frequency_hz);

// Obtenir le compteur de cycles
cycle_t get_cycle_count(void);

// Convertir cycles en microsecondes
uint32_t cycles_to_us(cycle_t cycles);

// Version simplifiée pour 32-bit
uint32_t cycles_to_us_simple(uint32_t cycles);

#endif // CYCLE_COUNT_H
