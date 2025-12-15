# ==================== CONFIGURATION ====================
CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy
SIZE = riscv64-unknown-elf-size
MKDIR = mkdir -p

# Librairie GCC pour opérations manquantes
LIBS = -lgcc

# ==================== DÉTECTION AUTOMATIQUE DOMAINE ====================
ifndef DOMAIN
  # Vérifier modèle ECG par nombre de paramètres
  ifneq ("$(wildcard firmware/model/arrays/conv1_weight_array.txt)","")
    CONV1_PARAMS := $(shell wc -w < firmware/model/arrays/conv1_weight_array.txt 2>/dev/null || echo 0)
    ifeq ($(CONV1_PARAMS),30)      # ECG: 30 paramètres conv1
      DOMAIN = medical
    else ifeq ($(CONV1_PARAMS),56) # Vibration: 56 paramètres
      DOMAIN = industrial
    else
      $(info Could not auto-detect domain from weights)
      $(info Please specify DOMAIN=medical or DOMAIN=industrial)
      $(error DOMAIN not specified)
    endif
  else
    $(info No weight files found)
    $(error Please specify DOMAIN=medical or DOMAIN=industrial)
  endif
endif

# Validation domaine
ifeq ($(DOMAIN),medical)
  PROJECT = ecg
  DOMAIN_FLAGS = -DECG_MODEL
  MODEL_CONFIG = Medical (ΔT ≤ 5°C)
else ifeq ($(DOMAIN),industrial)
  PROJECT = vibration
  DOMAIN_FLAGS = -DVIBRATION_MODEL
  MODEL_CONFIG = Industrial (ΔT ≤ 7°C)
else
  $(error DOMAIN must be 'medical' or 'industrial')
endif

# ==================== CONFIGURATION PLATEFORME ====================
ifndef PLATFORM
  PLATFORM = hifive1  # Défaut
endif

ifeq ($(PLATFORM),hifive1)
  ARCH = rv32imac
  ABI = ilp32
  PLATFORM_FLAGS = -DHIFIVE1 -DCPU_FREQ_MHZ=320 -DCPU_FREQ_HZ=32000000
  LDFLAGS += -T linker_hifive1.ld
  BUILD_DIR = build_hifive1
  # Pas de support float matériel
  ARCH_FLAGS = -march=rv32imac -mabi=ilp32
else ifeq ($(PLATFORM),k210)
  ARCH = rv64gc
  ABI = lp64
  PLATFORM_FLAGS = -DK210 -DCPU_FREQ_MHZ=400 -DCPU_FREQ_HZ=400000000
  LDFLAGS += -T linker_k210.ld
  CFLAGS += -mcmodel=medany
  LDFLAGS += -mcmodel=medany
  BUILD_DIR = build_k210
  ARCH_FLAGS = -march=rv64gc -mabi=lp64
else
  $(error PLATFORM must be hifive1 or k210)
endif

# ==================== FLAGS DE COMPILATION ====================
CFLAGS += $(ARCH_FLAGS)
CFLAGS += -Os -Wall -Wextra -Wno-unused-parameter
CFLAGS += -fno-common -ffreestanding -nostdlib -fno-builtin
CFLAGS += -fno-stack-protector
CFLAGS += -I. -Ifirmware/model -Iops -Iutils
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += $(DOMAIN_FLAGS)
CFLAGS += $(PLATFORM_FLAGS)

# Désactiver opérations float
CFLAGS += -mno-fdiv

# ==================== CONFIGURATION MODÈLE SPÉCIFIQUE ====================
ifeq ($(PROJECT),ecg)
  CFLAGS += -DINPUT_SIZE=360 -DCONV1_FILTERS=6 -DCONV2_FILTERS=12
  CFLAGS += -DLSTM_HIDDEN=24 -DSNN_HIDDEN=24 -DOUTPUT_SIZE=5
  CFLAGS += -DTIME_STEPS=32 -DFUSION_SIZE=24 -DFC1_SIZE=12
else ifeq ($(PROJECT),vibration)
  CFLAGS += -DINPUT_SIZE=1024 -DCONV1_FILTERS=8 -DCONV2_FILTERS=16
  CFLAGS += -DCONV3_FILTERS=32 -DLSTM_HIDDEN=32 -DOUTPUT_SIZE=4
  CFLAGS += -DTIME_STEPS=32 -DFC1_SIZE=64
endif

# ==================== FLAGS COMMUNS ====================
CFLAGS += -DQ_BITS=8 -DFIXED_SCALE=8 -DFIXED_SCALE_VAL=256
CFLAGS += -DENABLE_BENCHMARKING

# Flags de lien
LDFLAGS += -Wl,--gc-sections $(LIBS)

# ==================== FICHIERS SOURCES ====================
C_SRCS = main.c uart.c firmware/model/model.c firmware/model/model_weights.c \
         ops/math_ops.c utils/memutils.c utils/numutils.c utils/cycle_count.c \
         utils/thermal_manager.c
ASM_SRCS = start.S
C_OBJS = $(addprefix $(BUILD_DIR)/, $(C_SRCS:.c=.o))
ASM_OBJS = $(addprefix $(BUILD_DIR)/, $(ASM_SRCS:.S=.o))
OBJS = $(C_OBJS) $(ASM_OBJS)

# ==================== RÈGLES PRINCIPALES ====================
.PHONY: all clean size info thermal-demo help medical industrial

all: $(BUILD_DIR)/firmware.bin
	@echo ""
	@$(MAKE) --no-print-directory size

help:
	@echo "=== Hybrid SNN-QNN Build System ==="
	@echo "Usage: make PLATFORM=<platform> DOMAIN=<domain> [target]"
	@echo ""
	@echo "Platforms:"
	@echo "  PLATFORM=hifive1  - HiFive1 (RV32IMAC, 32MHz)"
	@echo "  PLATFORM=k210     - Kendryte K210 (RV64GC, 400MHz)"
	@echo ""
	@echo "Domains:"
	@echo "  DOMAIN=medical    - ECG arrhythmia (ΔT ≤ 5°C)"
	@echo "  DOMAIN=industrial - Vibration analysis (ΔT ≤ 7°C)"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build firmware (default)"
	@echo "  clean        - Clean build directory"
	@echo "  size         - Show memory usage"
	@echo "  info         - Show build configuration"
	@echo "  thermal-demo - Show thermal management info"
	@echo "  help         - This help message"
	@echo "  medical      - Build medical application"
	@echo "  industrial   - Build industrial application"
	@echo ""
	@echo "Examples:"
	@echo "  make medical                     # Build ECG for default platform"
	@echo "  make PLATFORM=k210 industrial    # Build vibration for K210"
	@echo "  make DOMAIN=medical clean all    # Clean and rebuild medical"

info:
	@echo "\n=== BUILD CONFIGURATION ==="
	@echo "Platform:    $(PLATFORM)"
	@echo "Architecture: $(ARCH)"
	@echo "Domain:      $(MODEL_CONFIG)"
	@echo "Project:     $(PROJECT)"
	@echo "Build dir:   $(BUILD_DIR)"
	@echo "Compiler:    $(CC)"
	@echo "CFLAGS:      $(CFLAGS)"

clean:
	@echo "Cleaning build directories..."
	@rm -rf build_hifive1 build_k210

size: $(BUILD_DIR)/firmware.elf
	@echo "\n=== MEMORY USAGE ==="
	@echo "Platform: $(PLATFORM)"
	@echo "Domain:   $(MODEL_CONFIG)"
	@echo "---------------------------------"
	@$(SIZE) $(BUILD_DIR)/firmware.elf
	@echo ""
	@echo "Thermal constraints:"
	@if [ "$(PROJECT)" = "ecg" ]; then \
		echo "  Medical: ΔT ≤ 5.0°C (patient safety)"; \
		echo "  Absolute: 50.0°C→4-bit, 70.0°C→2-bit"; \
		echo "  Standards: AAMI EC57:2025, ISO 13732-1"; \
	else \
		echo "  Industrial: ΔT ≤ 7.0°C (reliability)"; \
		echo "  Absolute: 50.0°C→4-bit, 70.0°C→2-bit"; \
		echo "  Standards: IEC 60529, MIL-STD-810"; \
	fi

thermal-demo:
	@echo "\n=== THERMAL MANAGEMENT SYSTEM ==="
	@echo "Domain: $(MODEL_CONFIG)"
	@echo ""
	@echo "ΔT CONSTRAINTS (domain-specific):"
	@if [ "$(PROJECT)" = "ecg" ]; then \
		echo "  Medical: ΔT ≤ 5.0°C"; \
		echo "  Rationale: Patient safety, skin temperature limits"; \
		echo "  Prevents: Thermal discomfort, burns (>41°C)"; \
		echo "  Standards: AAMI EC57:2025, ISO 13732-1"; \
	else \
		echo "  Industrial: ΔT ≤ 7.0°C"; \
		echo "  Rationale: Component reliability, MTBF optimization"; \
		echo "  Improves: Mean Time Between Failures (MTBF)"; \
		echo "  Standards: IEC 60529, MIL-STD-810"; \
	fi
	@echo ""
	@echo "ABSOLUTE PROTECTION (silicon safety):"
	@echo "  50.0°C → Switch from 8-bit to 4-bit (preventive)"
	@echo "  70.0°C → Switch from 4-bit to 2-bit (corrective)"
	@echo "  85.0°C → Emergency shutdown (safety)"
	@echo ""
	@echo "MANAGEMENT HIERARCHY:"
	@echo "  1. ΔT domain constraint (primary)"
	@echo "  2. Absolute temperature (secondary)"
	@echo "  3. Most restrictive applied"

# Règles de domaine spécifique
medical:
	$(MAKE) DOMAIN=medical $(MAKECMDGOALS)

industrial:
	$(MAKE) DOMAIN=industrial $(MAKECMDGOALS)

# ==================== RÈGLES DE COMPILATION ====================
$(BUILD_DIR)/%.o: %.c
	@$(MKDIR) $(dir $@)
	@echo "CC  $(notdir $<) [$(PROJECT)]"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.S
	@$(MKDIR) $(dir $@)
	@echo "AS  $(notdir $<)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/firmware.elf: $(OBJS)
	@echo "LD  $@"
	@$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@

$(BUILD_DIR)/firmware.bin: $(BUILD_DIR)/firmware.elf
	@echo "OBJCOPY $@"
	@$(OBJCOPY) -O binary $< $@
	@echo "\n✅ Build successful!"
	@echo "   Platform: $(PLATFORM)"
	@echo "   Domain: $(MODEL_CONFIG)"
	@echo "   Project: $(PROJECT)"
	@if [ "$(PROJECT)" = "ecg" ]; then \
		echo "   Model: ECG: 7.51KB, 97.49% accuracy"; \
	else \
		echo "   Model: Vibration: 16.82KB, 100% accuracy"; \
	fi
