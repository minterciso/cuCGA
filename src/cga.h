#ifndef __CGA_H
#define __CGA_H

#include <stdio.h>
#include "structs.h"

//ics holds the training ICs: params.n_train_ics shared by the whole population with
//params.shared_ics, params.n_train_ics per individual (individual i at i*n_train_ics) otherwise.
//They must be filled for the first generation; evolve() redraws them every generation after it.
void evolve(Individual *pop, Lattice *ics);
void crossOver(Individual *pop);
void mutate(Individual *pop, size_t amount);
//Decodes templates (if any), runs the CA on every individual's training ICs (laid out as in
//evolve()) and sets its fitness; lat and rules are scratch buffers of population*n_train_ics
//lattices and population rules
void evaluatePopulation(Individual *pop, const Lattice *ics, Lattice *lat, char *rules);
//Performance of rule on nICs unbiased (binomial density) ICs, as in the final evaluation of MCH/CMD.
//perf counts a uniform correct final state after CA_RUNS steps (the GA fitness criterion, whose
//step count can be random with --poisson-steps; the final evaluation always uses CA_RUNS);
//perfStrict additionally requires that state to be a fixed point of the rule.
void validateRule(const char *rule, int nICs, double *perf, double *perfStrict);

#endif //__CGA_H
