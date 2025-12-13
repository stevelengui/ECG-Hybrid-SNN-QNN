#ifndef UART_H
#define UART_H

#include <stdint.h>

// Configuration plateforme
#if defined(K210)
  #define UART_CLOCK_FREQ 400000000
#elif defined(HIFIVE1)
  #define UART_CLOCK_FREQ 16000000
#endif

// Interface de base
void uart_init(uint32_t baudrate);
void uart_putchar(char c);
void uart_puts(const char *s);
int uart_getchar(void);
int uart_available(void);

// Fonctions étendues
void uart_putint(int32_t val);
void uart_hexdump(uint32_t value);
void uart_printf(const char *fmt, ...);

// Debug (seulement compilé si UART_DEBUG=1)
#if UART_DEBUG
void uart_print_registers(void);
#define UART_PRINT_REG() uart_print_registers()
#else
#define UART_PRINT_REG() 
#endif

// Macros utiles
#define UART_PRINT_HEX(x) do { uart_puts("0x"); uart_hexdump(x); uart_puts("\n"); } while(0)

#endif
