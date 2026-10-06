#ifndef __BACKEND_H
#define __BACKEND_H

#include "structs.h"

//The only boundary between the GA and the hardware it runs the CA on: cga links a CPU
//implementation (backend_cpu.c), cuCga a CUDA one (kernel.cu). Everything else is shared.
#ifdef __cplusplus
extern "C" {
#endif

//Runs lat[t].steps steps (stopping early at a fixed point) on nLats lattices in place;
//lattice t uses rule t/latsPerRule
//(RULE_SIZE chars each) from rules
void runCA(Lattice *lat, const char *rules, int nLats, int latsPerRule);

#ifdef __cplusplus
}
#endif

#endif //__BACKEND_H
