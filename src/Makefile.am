.cu.o:
	$(NVCC) -o $@ -c $(NVCC_CFLAGS) --ptxas-options=-v --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" $<

CC = g++
CCLD=$(CC)
bin_PROGRAMS=cuCga
cuCga_SOURCES = main.c structs.h consts.h\
	        utils.h utils.c \
		ca.h ca.c \
                cga.h cga.cu

cuCga_CFLAGS  = $(CUDA_CFLAGS)
cuCga_LDADD   = $(CUDA_LIBS) 
cuCga_LDFLAGS = 
