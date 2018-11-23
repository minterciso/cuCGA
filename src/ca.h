#ifndef __CA_H
#define __CA_H

#include <stdio.h>

#include "structs.h"

void createRandomDens(int *dens);
int createRandomLattice(char *lattice, int dens, int binomial);
void createRandomRules(Individual *population);
void testCga(Individual *ind, Lattice *lat);


#endif //__CA_H

