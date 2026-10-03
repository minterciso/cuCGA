#ifndef __KERNEL_H
#define __KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "structs.h"

__global__ void executeCAKernel(Lattice *lat, const char *rules, int nLats, int latsPerRule);

#endif //__KERNEL_H

