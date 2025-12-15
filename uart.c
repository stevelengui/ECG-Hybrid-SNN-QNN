#include "uart.h"
#include <stdarg.h>

// Structure des registres UART
typedef struct {
    volatile uint32_t TXDATA;
    volatile uint32_t RXDATA;
    volatile uint32_t TXCTRL;
    volatile uint32_t RXCTRL;
    volatile uint32_t IE;
    volatile uint32_t IP;
    volatile uint32_t DIV;
} UART_Registers;

#if defined(HIFIVE1)
  #define UART_BASE 0x10013000
#elif defined(K210)
  #define UART_BASE 0x38000000
#endif

static UART_Registers *const uart = (UART_Registers *)UART_BASE;

void uart_init(uint32_t baudrate) {
    uart->IE = 0; // Disable interrupts
    uart->DIV = (UART_CLOCK_FREQ / baudrate) - 1;
    uart->TXCTRL = 0x1; // Enable TX
    uart->RXCTRL = 0x1; // Enable RX
}

void uart_putchar(char c) {
    while (uart->TXDATA & 0x80000000); // Wait for TX ready
    uart->TXDATA = c;
    
    if (c == '\n') { // Auto CR for LF
        while (uart->TXDATA & 0x80000000);
        uart->TXDATA = '\r';
    }
}

void uart_puts(const char *s) {
    while (*s) uart_putchar(*s++);
}

int uart_getchar(void) {
    while (!(uart->RXDATA & 0x80000000)); // Wait for data
    return uart->RXDATA & 0xFF;
}

int uart_available(void) {
    return !(uart->RXDATA & 0x80000000);
}

// uart_putint - version optimisée pour embedded
void uart_putint(int32_t val) {
    char buffer[12];
    int i = 0;
    int is_neg = val < 0;
    
    if (is_neg) val = -val;
    if (val == 0) buffer[i++] = '0';
    
    while (val > 0 && i < 11) {
        buffer[i++] = '0' + (val % 10);
        val /= 10;
    }
    
    if (is_neg) uart_putchar('-');
    while (i > 0) uart_putchar(buffer[--i]);
}

// uart_putfloat - SUPPRIMÉ (déplacé à numutils.c)
// void uart_putfloat(float value, uint8_t decimals) {
//     ...
// }

void uart_hexdump(uint32_t value) {
    const char hex[] = "0123456789ABCDEF";
    for (int i = 28; i >= 0; i -= 4) {
        uart_putchar(hex[(value >> i) & 0xF]);
    }
}

void uart_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    while (*fmt) {
        if (*fmt == '%') {
            switch (*++fmt) {
                case 'd': uart_putint(va_arg(args, int32_t)); break;
                case 'x': uart_hexdump(va_arg(args, uint32_t)); break;
                case 's': uart_puts(va_arg(args, char*)); break;
                // %f supprimé car nécessite opérations flottantes
                default: uart_putchar(*fmt); break;
            }
        } else {
            uart_putchar(*fmt);
        }
        fmt++;
    }
    va_end(args);
}

#if defined(UART_DEBUG) && UART_DEBUG
void uart_print_registers(void) {
    uart_puts("\nUART Registers:\n");
    uart_puts("TXDATA: 0x"); uart_hexdump(uart->TXDATA); uart_puts("\n");
    uart_puts("RXDATA: 0x"); uart_hexdump(uart->RXDATA); uart_puts("\n");
    uart_puts("TXCTRL: 0x"); uart_hexdump(uart->TXCTRL); uart_puts("\n");
    uart_puts("DIV:    0x"); uart_hexdump(uart->DIV); uart_puts("\n");
}
#endif
