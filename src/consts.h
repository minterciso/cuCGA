#ifndef __CONSTS_H
#define __CONSTS_H

//#define DEBUG
//#define VALIDATE

//Structural definition (Production run)
#ifndef DEBUG

#ifdef VALIDATE
#define LAT_SIZE 149
#define MAX_LATS 1000
#endif

#ifndef VALIDATE
#define LAT_SIZE 149
#define MAX_LATS 100
#endif

#define RADIUS 3
#define RULE_SIZE 128
#endif

//Structural definition (Debug run)
#ifdef DEBUG

#ifdef VALIDATE
#define LAT_SIZE 149
#define MAX_LATS 1000
#endif

#ifndef VALIDATE
#define LAT_SIZE 149
#define MAX_LATS 10
#endif

#define RADIUS 3
#define RULE_SIZE 128
#define FNAME_SIZE 30
#define FNAME "logs/individualXXX_XXX.log"
#endif

//Ternary representation (templates)
#define NEIGH_SIZE (RADIUS*2+1)
#define MAX_TEMPLATES 18 //Largest T_max probed in the paper

//Runnable definition
#define CA_RUNS 300
#define DEFAULT_GENERATIONS 100 //Generations per GA run (--generations)

//GA Probabilities (and elitism amount)
//Population, elite, crossover rate, mutation rate and the representation are runtime parameters (see params.h);
//these are only their defaults.
#define DEFAULT_POPULATION 100 //P in MCH/CMD (--population)
#define DEFAULT_ELITE_PCT 20.0 //E = 20 of P = 100 (--elite)
#define DEFAULT_CROSS_RATE 1.0  //CMD: p_c = 100% (MCH uses 0.8)
#define DEFAULT_MUT_RATE 0.016 //CMD: 1.6% per bit, ~2 bits per individual
#define DEFAULT_TPL_MUT_RATE 0.064 //Templates: ~2 mutations per individual at t_max=9 (4.5 templates * 7 cells)

//Rule representation and template parameters (see params.h)
#define DEFAULT_T_MAX 9
#define DEFAULT_HASH_PROB (2.0/7.0) //~2 '#' per 7-cell template

//Best rule found in original CGA paper
//#define USE_BEST
#define BEST_CGA "0504058605000f77037755877bffb77f"
//#define BEST_CGA "100111215030114d01613507143b05bf"

//File output: per-generation trace, written by evolve() only (single writer)
#define F_OUTPUT
#define F_OUTPUT_FILE "logs/output.log"

//Final evaluation of the best rule (MCH/CMD): binomially distributed ICs
#define DEFAULT_N_ICS 10000

#endif //__CONSTS_H
