#ifndef __CGA_H
#define __CGA_H

#include <stdio.h>
#include "structs.h"

void evolve(Individual *pop);
void crossOver(Individual *pop);
void mutate(Individual *pop, size_t amount);
//Decodes templates (if any), runs the CA on every individual's lattices and sets its fitness;
//lat and rules are scratch buffers of POPULATION*MAX_LATS lattices and POPULATION rules
void evaluatePopulation(Individual *pop, Lattice *lat, char *rules);
//Performance of rule on nICs unbiased (binomial density) ICs, as in the final evaluation of MCH/CMD.
//perf counts a uniform correct final state after CA_RUNS steps (same criterion as the GA fitness);
//perfStrict additionally requires that state to be a fixed point of the rule.
void validateRule(const char *rule, int nICs, double *perf, double *perfStrict);

#endif //__CGA_H
