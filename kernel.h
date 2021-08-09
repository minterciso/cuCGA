#ifndef __KERNEL_H
#define __KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "structs.h"

__device__ void d_dec2bin(int decimal, char *bin, int size);
__device__ int  d_bin2dec(char *bin, int size);
__device__ void d_hex2bin(char *hex, char *bin, int h_size, int b_size);
__device__ void d_bin2hex(char *hex, char *bin, int h_size, int b_size);

__global__ void executeCA(Lattice *lat, char *rule);

#endif //__KERNEL_H

