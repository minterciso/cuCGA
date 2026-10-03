#ifndef __PARAMS_H
#define __PARAMS_H

#include <stdio.h>
#include <limits.h>
#include "consts.h"

//Runtime parameters. Filled once by parseParams() before any thread starts,
//read-only afterwards.
typedef enum Representation
{
  REP_BINARY, //128-bit rule string
  REP_SINGLE, //Templates, orientation always '1'
  REP_DOUBLE  //Templates, random orientation per individual
}Representation;

typedef struct Params
{
  double mut_rate;               //Per-symbol mutation probability [0,1]
  double cross_rate;             //Crossover probability p_c [0,1]
  Representation representation;
  int t_max;                     //Maximum initial templates per individual [0,MAX_TEMPLATES]
  double hash_prob;              //Probability of '#' in each template cell [0,1]
  int generations;               //Generations of the GA (G in MCH/CMD)
  int population;                //Population size P [2,MAX_POPULATION]
  double elite_pct;              //Elite as a percentage of the population (0,100)
  int elite;                     //Elite size E = round(population*elite_pct/100), in [1,population-1]
  unsigned int seed;             //srand() seed; taken from the clock unless --seed is given
  int n_ics;                     //Binomial ICs for the final evaluation of a rule
  const char *validate_hex;      //When set, only evaluate this rule (hex, MCH order), no GA
  const char *csv_path;          //When set, per-generation fitness statistics are written there as CSV
}Params;

extern Params params;

//Returns 0 to continue, 1 if the program should exit successfully (--help), -1 on error
int parseParams(int argc, char *argv[]);
void printParams(FILE *stream);

//Largest population whose lattices can still be indexed with an int (population*MAX_LATS)
#define MAX_POPULATION (INT_MAX/MAX_LATS)

#endif //__PARAMS_H
