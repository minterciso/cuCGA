CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc

# GPU architecture: 'native' targets the GPU(s) present on this machine.
# Override for other targets, e.g. make ARCH=sm_89
ARCH ?= native

# nvcc only supports a bounded range of host GCC versions (CUDA 13.3: <= 15).
# Prefer a compat compiler if installed (Fedora: dnf install gcc15-c++),
# otherwise fall back to the system g++ and skip nvcc's version check.
HOST_CXX ?= $(firstword $(wildcard /usr/bin/g++-15 /usr/bin/g++-14 /usr/bin/g++-13))
ifneq ($(HOST_CXX),)
CCBIN = -ccbin $(HOST_CXX)
else
CCBIN = -allow-unsupported-compiler
endif

CFLAGS  = -O2 -arch=$(ARCH) $(CCBIN)
DFLAGS  = -O0 -g -G --ptxas-options=-v -arch=$(ARCH) $(CCBIN)

SRCS = main.cu utils.cu ca.cu kernel.cu cga.cu
HDRS = ca.h cga.h consts.h kernel.h structs.h utils.h

all: cuCga

cuCga: $(SRCS) $(HDRS) | logs
	$(NVCC) $(CFLAGS) -o $@ $(SRCS)

debug: $(SRCS) $(HDRS) | logs
	$(NVCC) $(DFLAGS) -o cuCga $(SRCS)

logs:
	mkdir -p logs

run: cuCga
	./cuCga

clean:
	rm -f cuCga

.PHONY: all debug run clean
