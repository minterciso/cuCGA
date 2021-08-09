#ifndef __CGA_H
#define __CGA_H

#include <stdio.h>
#include "structs.h"

//void evolve(Individual *pop);
void crossOver(Individual *pop);
void mutate(Individual *pop, size_t amount);

#endif //__CGA_H

