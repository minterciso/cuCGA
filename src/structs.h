#ifndef __STRUCTS_H
#define __STRUCTS_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "consts.h"

typedef struct Lattice
{
  char cells[LAT_SIZE];
  unsigned int density;
}Lattice;

//Ternary representation (de Oliveira & Interciso, CEC 2011): a rule is a set of
//templates over {0,1,#}, one symbol per neighbourhood cell, '#' meaning "don't care".
//A neighbourhood matched by any template maps to the individual's orientation bit,
//every other neighbourhood to its complement.
typedef struct Template
{
  char cells[NEIGH_SIZE]; //cells[0] = s(i-RADIUS) ... cells[NEIGH_SIZE-1] = s(i+RADIUS)
}Template;

typedef struct Individual
{
  Lattice lat[MAX_LATS];
  char rule[RULE_SIZE];        //Binary LUT run by the CA; decoded from tpl[] when using templates
  Template tpl[MAX_TEMPLATES]; //Genotype for the single/double representations
  int n_tpl;                   //Fixed at creation: crossover swaps templates one for one
  char orientation;            //Output bit ('0'/'1') of the neighbourhoods matched by tpl[]
  unsigned int fitness;
  int id;
}Individual;

#endif //__STRUCTS_H
