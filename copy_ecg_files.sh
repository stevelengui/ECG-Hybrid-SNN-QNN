#!/bin/bash
echo "=== COPYING ECG FILES ==="

SOURCE_DIR=~/riscv-project
DEST_DIR=~/ECG-Hybrid-SNN-QNN

# Fichiers racine
cp $SOURCE_DIR/main.c $DEST_DIR/
cp $SOURCE_DIR/Makefile $DEST_DIR/
cp $SOURCE_DIR/uart.c $DEST_DIR/
cp $SOURCE_DIR/uart.h $DEST_DIR/
cp $SOURCE_DIR/start.S $DEST_DIR/
cp $SOURCE_DIR/linker_hifive1.ld $DEST_DIR/
cp $SOURCE_DIR/linker_k210.ld $DEST_DIR/

# Firmware
cp $SOURCE_DIR/firmware/model/model.c $DEST_DIR/firmware/model/

# Ops
cp $SOURCE_DIR/ops/math_ops.c $DEST_DIR/ops/
cp $SOURCE_DIR/ops/math_ops.h $DEST_DIR/ops/

# Utils
cp $SOURCE_DIR/utils/thermal_manager.h $DEST_DIR/utils/
cp $SOURCE_DIR/utils/thermal_manager.c $DEST_DIR/utils/
cp $SOURCE_DIR/utils/memutils.c $DEST_DIR/utils/
cp $SOURCE_DIR/utils/memutils.h $DEST_DIR/utils/
cp $SOURCE_DIR/utils/numutils.c $DEST_DIR/utils/
cp $SOURCE_DIR/utils/numutils.h $DEST_DIR/utils/
cp $SOURCE_DIR/utils/cycle_count.c $DEST_DIR/utils/
cp $SOURCE_DIR/utils/cycle_count.h $DEST_DIR/utils/

echo "✅ All ECG files copied successfully!"
