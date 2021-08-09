NVCC=`which nvcc`
CFLAGS=-O0 --ptxas-options=-v -g 
DFLAGS=-DDEBUG -deviceemu

all: cuCga

cuCga: main.cu utils.cu ca.cu kernel.cu cga.cu
	${NVCC} ${CFLAGS} -o cuCga main.cu utils.cu ca.cu kernel.cu cga.cu

clean:
	rm cuCga
