#include "memutils.h"
#include <stddef.h>
#include <stdint.h>

// Section spéciale pour garantir la présence
__attribute__((section(".text.memutils"), used, noinline))
void* memcpy(void* dest, const void* src, size_t n) {
    if (n == 0 || dest == src) return dest;
    
    uint8_t *d = dest;
    const uint8_t *s = src;
    
    // Copie mot (32-bit) si aligné
    if ((((uintptr_t)d | (uintptr_t)s) & 0x3) == 0) {
        while (n >= 4) {
            *((uint32_t*)d) = *((const uint32_t*)s);
            d += 4;
            s += 4;
            n -= 4;
        }
    }
    
    // Copie octets restants
    while (n--) *d++ = *s++;
    
    return dest;
}

__attribute__((section(".text.memutils"), used, noinline))
void* memset(void* ptr, int value, size_t num) {
    if (num == 0) return ptr;
    
    uint8_t *p = ptr;
    uint8_t v = (uint8_t)value;
    
    // Remplissage par mot (32-bit) si aligné
    if (((uintptr_t)p & 0x3) == 0) {
        uint32_t pattern = v | (v << 8) | (v << 16) | (v << 24);
        while (num >= 4) {
            *((uint32_t*)p) = pattern;
            p += 4;
            num -= 4;
        }
    }
    
    // Octets restants
    while (num--) *p++ = v;
    
    return ptr;
}

void* memmove(void* dest, const void* src, size_t n) {
    if (dest == src || n == 0) return dest;
    
    uint8_t *d = dest;
    const uint8_t *s = src;
    
    if (d < s) {
        return memcpy(d, s, n);
    } else {
        d += n;
        s += n;
        while (n--) *--d = *--s;
    }
    return dest;
}

int memcmp(const void* ptr1, const void* ptr2, size_t n) {
    const uint8_t *p1 = ptr1;
    const uint8_t *p2 = ptr2;
    
    while (n--) {
        if (*p1 != *p2) {
            return (*p1 < *p2) ? -1 : 1;
        }
        p1++;
        p2++;
    }
    return 0;
}
