MEMORY {
    FLASH (rx) : ORIGIN = 0x20400000, LENGTH = 512K
    RAM (rwx)  : ORIGIN = 0x80000000, LENGTH = 16K
}

ENTRY(_start)

SECTIONS {
    .text : {
        *(.text.init)
        *(.text .text.*)
        *(.rodata .rodata.*)
        . = ALIGN(4);
    } > FLASH

    .data : {
        *(.data .data.*)
        . = ALIGN(4);
    } > RAM AT > FLASH

    .bss : {
        _sbss = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    .stack : {
        . = ALIGN(4);
        _stack_end = .;
        . += DEFINED(__stack_size) ? __stack_size : 4K;
        _stack_top = .;
    } > RAM

    /DISCARD/ : {
        *(.eh_frame)
    }
}
