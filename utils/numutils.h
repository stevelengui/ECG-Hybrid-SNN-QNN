#ifndef NUMUTILS_H
#define NUMUTILS_H

#include <stdint.h>

// Simple PRNG for freestanding environment
uint32_t simple_rand(void);

// Fixed-point math utilities
int32_t fp_multiply(int32_t a, int32_t b, uint8_t shift);
int32_t fp_divide(int32_t a, int32_t b, uint8_t shift);

// Utility functions for UART output
void uart_puthex(uint32_t value);
void uart_putfloat(float value, uint8_t decimals);

#endif // NUMUTILS_H
