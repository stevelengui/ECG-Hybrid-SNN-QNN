.section .text.init
.global _start
.option push
.option norelax

_start:
    la sp, _stack_top
    
    /* Clear BSS section */
    la a0, _sbss
    la a1, _ebss
    bgeu a0, a1, mem_clear_done

mem_clear_loop:
#if __riscv_xlen == 32
    sw zero, 0(a0)
    addi a0, a0, 4
#else
    sd zero, 0(a0)
    addi a0, a0, 8
#endif
    bltu a0, a1, mem_clear_loop

mem_clear_done:
    call main
    j .

.section .stack
.align 4
_stack_top:
/* L'espace stack est alloué par le linker script */
.option pop
