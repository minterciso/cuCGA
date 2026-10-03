#ifndef __TEMPLATES_H
#define __TEMPLATES_H

#include "structs.h"

//Random genotype: n_tpl uniform in [0,t_max], each cell '#' with probability
//hash_prob, otherwise '0'/'1'. Orientation is '1' (single) or random (double).
void createRandomTemplates(Individual *ind);
//Writes the binary LUT (ind->rule) that the templates and orientation encode
void decodeTemplates(Individual *ind);
//Swaps one randomly chosen template of a with one of b; both need n_tpl >= 1
void swapTemplates(Individual *a, Individual *b);
//Each template cell mutates with probability mut_rate to one of the other two symbols
void mutateTemplates(Individual *ind);

#endif //__TEMPLATES_H
