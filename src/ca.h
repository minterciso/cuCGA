#ifndef __CA_H
#define __CA_H

#include <stdio.h>
#include "structs.h"
#include "consts.h"

//n training ICs: density uniform on [0,LAT_SIZE] (binomial with VALIDATE); each runs CA_RUNS
//steps, or a Poisson(POISSON_STEPS_MEAN) number of steps with --poisson-steps
void createTrainingLattices(Lattice *lat, int n);
void createRandomRules(Individual *ind);
//Unbiased ICs: each cell is 1 with probability 0.5, so the density is Binomial(LAT_SIZE,0.5); CA_RUNS steps
void createUnbiasedLattices(Lattice *lat, int n);
//1 if the final lattice is uniform with the majority bit of its IC, 0 otherwise; *fixed is set
//to whether that uniform state is also a fixed point of the rule
int classify(const Lattice *lat, const char *rule, int *fixed);

//CPU reference implementation of the CA
void executeCA(Lattice *lat, const char *rule, int ind_idx, int th_idx);
void cpuRunCA(Lattice *lat, const char *rules, int nLats, int latsPerRule);
void printCA(FILE *stream, Lattice *lat, int mode);

#endif //__CA_H
