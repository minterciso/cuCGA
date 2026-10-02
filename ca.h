#ifndef __CA_H
#define __CA_H

#include <stdio.h>

#include "structs.h"

void createRandomLattice(Individual *ind);
void createRandomRules(Individual *ind);
void createUnbiasedLattices(Lattice *lat, int n);

#endif //__CA_H

