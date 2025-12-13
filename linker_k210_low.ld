MEMORY {
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 6M
}

SECTIONS {
    .text : {
        *(.text.start)
        *(.text .text.*)
    } > RAM AT> RAM
    
    . = ALIGN(4);
    
    .rodata : {
        *(.rodata .rodata.*)
    } > RAM
    
    .data : {
        *(.data .data.*)
    } > RAM
    
    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM
    
    /DISCARD/ : {
        *(.comment)
        *(.note*)
    }
    
    /* Stack au début de la RAM pour faciliter l'adressage */
    _sp = 0x80000000 + 0x1000;
}
