#!/bin/bash
echo "=== BUILDING FOR K210 (RV64) ==="

# Clean
rm -rf build
mkdir -p build

# Compiler avec médium model et relaxation
riscv64-unknown-elf-gcc -DK210 -DCPU_FREQ_MHZ=400 -march=rv64gc -mabi=lp64 \
  -mcmodel=medany -mrelax \
  -Os -Wall -Wextra -w \
  -fno-common -ffreestanding -nostdlib -fno-builtin \
  -I. -Ifirmware/model -Iops -Iutils \
  -fdata-sections -ffunction-sections \
  -DINPUT_SIZE=360 -DCONV1_FILTERS=6 -DCONV2_FILTERS=12 \
  -DLSTM_HIDDEN=24 -DSNN_HIDDEN=24 -DOUTPUT_SIZE=5 \
  -DTIME_STEPS=32 -DQ_BITS=8 -DFIXED_SCALE=8 \
  -DENABLE_BENCHMARKING \
  -c main.c -o build/main.o

# Compiler les autres fichiers avec les mêmes flags
for file in uart.c firmware/model/model.c firmware/model/model_weights.c \
            ops/math_ops.c utils/memutils.c utils/numutils.c utils/cycle_count.c; do
    riscv64-unknown-elf-gcc -DK210 -DCPU_FREQ_MHZ=400 -march=rv64gc -mabi=lp64 \
      -mcmodel=medany -mrelax \
      -Os -Wall -Wextra -w \
      -fno-common -ffreestanding -nostdlib -fno-builtin \
      -I. -Ifirmware/model -Iops -Iutils \
      -fdata-sections -ffunction-sections \
      -DINPUT_SIZE=360 -DCONV1_FILTERS=6 -DCONV2_FILTERS=12 \
      -DLSTM_HIDDEN=24 -DSNN_HIDDEN=24 -DOUTPUT_SIZE=5 \
      -DTIME_STEPS=32 -DQ_BITS=8 -DFIXED_SCALE=8 \
      -DENABLE_BENCHMARKING \
      -c $file -o build/$(basename $file .c).o
done

# Compiler start.S
riscv64-unknown-elf-gcc -DK210 -DCPU_FREQ_MHZ=400 -march=rv64gc -mabi=lp64 \
  -mcmodel=medany -mrelax \
  -Os -Wall -Wextra -w \
  -fno-common -ffreestanding -nostdlib -fno-builtin \
  -I. -Ifirmware/model -Iops -Iutils \
  -c start.S -o build/start.o

# Linker
riscv64-unknown-elf-gcc -DK210 -DCPU_FREQ_MHZ=400 -march=rv64gc -mabi=lp64 \
  -mcmodel=medany -mrelax \
  -T linker_k210.ld -Wl,--gc-sections \
  build/*.o build/*/*.o build/*/*/*.o 2>/dev/null \
  -o build/firmware.elf

# Vérifier si ça a réussi
if [ $? -eq 0 ]; then
    riscv64-unknown-elf-objcopy -O binary build/firmware.elf build/firmware.bin
    echo "✅ SUCCESS - K210 build"
    echo "Size: $(wc -c < build/firmware.bin) bytes"
else
    echo "❌ Link failed, trying alternative..."
    # Essayer sans médium model
    riscv64-unknown-elf-gcc -DK210 -march=rv64gc -mabi=lp64 \
      -T linker_k210.ld -Wl,--gc-sections -nostdlib \
      build/*.o build/*/*.o build/*/*/*.o 2>/dev/null \
      -o build/firmware.elf
    if [ $? -eq 0 ]; then
        riscv64-unknown-elf-objcopy -O binary build/firmware.elf build/firmware.bin
        echo "✅ SUCCESS - K210 build (no medany)"
        echo "Size: $(wc -c < build/firmware.bin) bytes"
    else
        echo "❌ Build failed"
        exit 1
    fi
fi
