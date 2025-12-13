MEMORY {
    RAM (rwx) : ORIGIN = 0x80020000, LENGTH = 6M
}

ENTRY(_start)

SECTIONS {
    .text : {
        *(.text.init)
        *(.text .text.*)
        . = ALIGN(8);
    } > RAM

    .rodata : {
        *(.rodata .rodata.*)
        . = ALIGN(8);
    } > RAM

    .data : {
        *(.data .data.*)
        . = ALIGN(8);
    } > RAM

    .bss : {
        _sbss = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(8);
        _ebss = .;
    } > RAM

    .stack : {
        . = ALIGN(16);
        _stack_end = .;
        . += DEFINED(__stack_size) ? __stack_size : 16K;
        _stack_top = .;
    } > RAM

    /DISCARD/ : {
        *(.eh_frame)
    }
}
