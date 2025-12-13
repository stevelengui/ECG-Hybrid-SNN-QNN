#ifndef MEMUTILS_H
#define MEMUTILS_H

#include <stddef.h>

// Déclarations FORCEES
__attribute__((used, noinline, visibility("default")))
void* memcpy(void* dest, const void* src, size_t n);

__attribute__((used, noinline, visibility("default")))
void* memset(void* ptr, int value, size_t num);

// Fonctions standard
void* memmove(void* dest, const void* src, size_t n);
int memcmp(const void* ptr1, const void* ptr2, size_t n);

#endif
