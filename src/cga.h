#ifndef __CGA_H
#define __CGA_H

#include <stdio.h>
#include <stdlib.h>
#include "structs.h"

int cga(Individual *h_pop, Lattice* h_lat);
void testCga(Individual *h_pop, Lattice* h_lat);
void crossOver(Individual *h_pop);
void mutate(Individual *h_pop);

#endif //__CGA_H

