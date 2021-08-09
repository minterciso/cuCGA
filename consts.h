#ifndef __CONSTS_H
#define __CONSTS_H

#define DEBUG
//#define VALIDATE

//Structural definition (Production run)
#ifndef DEBUG
#define POPULATION 100

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
#define POPULATION 100

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
#define FNAME_SIZE 25
#define FNAME "logs/individualXXX.log"
#endif

//Runnable definition
#define CA_RUNS 300
#define GA_RUNS 100

//GA Probabilities (and elitism amount)
#define CROSS_AMOUNT 20
#define CROSS_RATE 10
#define MUT_RATE 0.016

//Define the mode we are studying (binary or ternary representation)
#define MODE 2

//Best rule found in original CGA paper
//#define USE_BEST
//#define BEST_CGA "0504058605000f77037755877bffb77f"
//#define BEST_CGA "100111215030114d01613507143b05bf"

//File output
#define F_OUTPUT
#define F_OUTPUT_FILE "logs/output.log"

#endif //__CONSTS_H
