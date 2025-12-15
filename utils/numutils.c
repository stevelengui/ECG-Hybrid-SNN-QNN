#include "numutils.h"
#include "uart.h"
#include <stdint.h>

// Simple PRNG state
static uint32_t prng_state = 123456789;

uint32_t simple_rand(void) {
    prng_state = prng_state * 1103515245 + 12345;
    return (prng_state >> 16) & 0x7FFF;
}

int32_t fp_multiply(int32_t a, int32_t b, uint8_t shift) {
    int64_t result = (int64_t)a * (int64_t)b;
    return (int32_t)(result >> shift);
}

int32_t fp_divide(int32_t a, int32_t b, uint8_t shift) {
    if (b == 0) return 0;
    int64_t result = ((int64_t)a << shift) / (int64_t)b;
    return (int32_t)result;
}

void uart_puthex(uint32_t value) {
    const char hex_digits[] = "0123456789ABCDEF";
    char buffer[9];
    
    for (int i = 7; i >= 0; i--) {
        buffer[i] = hex_digits[value & 0xF];
        value >>= 4;
    }
    buffer[8] = '\0';
    
    uart_puts("0x");
    uart_puts(buffer);
}

// Version simplifiée de uart_putfloat (éviter les opérations flottantes)
void uart_putfloat(float value, uint8_t decimals) {
    // Convertir en entiers pour l'affichage
    int int_part = (int)value;
    int sign = 1;
    
    if (value < 0) {
        sign = -1;
        int_part = -int_part;
        uart_putchar('-');
    }
    
    // Partie entière
    uart_putint(int_part);
    
    // Partie décimale (simplifiée)
    if (decimals > 0) {
        uart_putchar('.');
        
        // Calculer la partie décimale sans opérations flottantes complexes
        float frac = value - (float)(int_part * sign);
        if (frac < 0) frac = -frac;
        
        for (uint8_t d = 0; d < decimals; d++) {
            frac = frac * 10.0f;
            int digit = (int)frac;
            uart_putint(digit);
            frac = frac - (float)digit;
        }
    }
}
