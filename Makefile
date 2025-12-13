CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy
SIZE = riscv64-unknown-elf-size

ifeq ($(PLATFORM),hifive1)
  ARCH = rv32imac
  ABI = ilp32
  CFLAGS += -DHIFIVE1 -DCPU_FREQ_MHZ=320
  LDFLAGS += -T linker_hifive1.ld
else ifeq ($(PLATFORM),k210)
  ARCH = rv64gc
  ABI = lp64
  CFLAGS += -DK210 -DCPU_FREQ_MHZ=400
  LDFLAGS += -T linker_k210.ld
  # Essentiel pour K210 - modèle de code médium
  CFLAGS += -mcmodel=medany
  LDFLAGS += -mcmodel=medany
else
  $(error PLATFORM must be hifive1 or k210)
endif

CFLAGS += -march=$(ARCH) -mabi=$(ABI)
CFLAGS += -Os -Wall -Wextra -w
CFLAGS += -fno-common -ffreestanding -nostdlib -fno-builtin -fno-stack-protector
CFLAGS += -I. -Ifirmware/model -Iops -Iutils
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += -DINPUT_SIZE=360 -DCONV1_FILTERS=6 -DCONV2_FILTERS=12
CFLAGS += -DLSTM_HIDDEN=24 -DSNN_HIDDEN=24 -DOUTPUT_SIZE=5
CFLAGS += -DTIME_STEPS=32 -DQ_BITS=8 -DFIXED_SCALE=8
CFLAGS += -DENABLE_BENCHMARKING

LDFLAGS += -Wl,--gc-sections

BUILD_DIR = build
C_SRCS = main.c uart.c firmware/model/model.c firmware/model/model_weights.c \
         ops/math_ops.c utils/memutils.c utils/numutils.c utils/cycle_count.c
ASM_SRCS = start.S
C_OBJS = $(addprefix $(BUILD_DIR)/, $(C_SRCS:.c=.o))
ASM_OBJS = $(addprefix $(BUILD_DIR)/, $(ASM_SRCS:.S=.o))
OBJS = $(C_OBJS) $(ASM_OBJS)

.PHONY: all clean size

all: $(BUILD_DIR)/firmware.bin size

clean:
	@rm -rf $(BUILD_DIR)

size: $(BUILD_DIR)/firmware.elf
	@echo "\n=== Memory Usage ==="
	@echo "Platform: $(PLATFORM)"
	@echo "Arch: $(ARCH)"
	@echo "-------------------------"
	@$(SIZE) $(BUILD_DIR)/firmware.elf
	@echo "Binary: $$(wc -c < $(BUILD_DIR)/firmware.bin) bytes"

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo "CC  $(notdir $<)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.S
	@mkdir -p $(dir $@)
	@echo "AS  $(notdir $<)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/firmware.elf: $(OBJS)
	@echo "LD  $@"
	@$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@

$(BUILD_DIR)/firmware.bin: $(BUILD_DIR)/firmware.elf
	@echo "OBJCOPY $@"
	@$(OBJCOPY) -O binary $< $@
	@echo "\n✅ Build successful for $(PLATFORM)"
