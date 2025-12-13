#include "numutils.h"
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
    // Cette fonction utilise uart_putchar, donc elle reste dans uart.c
    // ou on peut la laisser ici si elle utilise une fonction publique
    const char hex_digits[] = "0123456789ABCDEF";
    char buffer[9];
    buffer[8] = '\0';
    
    for (int i = 7; i >= 0; i--) {
        buffer[i] = hex_digits[value & 0xF];
        value >>= 4;
    }
    
    // Ces fonctions doivent être définies dans uart.c
    extern void uart_puts(const char *s);
    uart_puts("0x");
    uart_puts(buffer);
}

void uart_putfloat(float value, uint8_t decimals) {
    // Convert float to integer part
    int32_t int_part = (int32_t)value;
    
    // Ces fonctions doivent être définies dans uart.c
    extern void uart_putint(int32_t value);
    extern void uart_puts(const char *s);
    
    uart_putint(int_part);
    
    if (decimals > 0) {
        uart_puts(".");
        
        // Get fractional part
        float frac = value - int_part;
        if (frac < 0) frac = -frac;
        
        for (uint8_t i = 0; i < decimals; i++) {
            frac *= 10;
            int digit = (int)frac;
            uart_putint(digit);
            frac -= digit;
        }
    }
}
