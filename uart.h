#ifndef UART_H
#define UART_H

#include <stdint.h>

// Configuration plateforme
#if defined(K210)
  #define UART_CLOCK_FREQ 400000000
#elif defined(HIFIVE1)
  #define UART_CLOCK_FREQ 16000000
#else
  #define UART_CLOCK_FREQ 16000000  // Valeur par défaut
#endif

// Interface de base
void uart_init(uint32_t baudrate);
void uart_putchar(char c);
void uart_puts(const char *s);
int uart_getchar(void);
int uart_available(void);

// Fonctions étendues
void uart_putint(int32_t val);
void uart_putfloat(float value, uint8_t decimals);
void uart_hexdump(uint32_t value);
void uart_printf(const char *fmt, ...);

// Alias pour compatibilité
#define uart_putc uart_putchar
#define uart_getc uart_getchar

// Debug (seulement compilé si UART_DEBUG=1)
#if defined(UART_DEBUG) && UART_DEBUG
void uart_print_registers(void);
#define UART_PRINT_REG() uart_print_registers()
#else
#define UART_PRINT_REG() ((void)0)
#endif

// Macros utiles
#define UART_PRINT_HEX(x) do { uart_puts("0x"); uart_hexdump(x); uart_puts("\n"); } while(0)

#endif // UART_H
