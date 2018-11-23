#ifndef __STRUCTS_H
#define __STRUCTS_H

#include <stdio.h>
#include "consts.h"

typedef struct Lattice
{
  char cells[LAT_SIZE];
  unsigned int density;
}Lattice;

typedef struct Individual
{
  char rule[RULE_SIZE+1];
  unsigned int fitness;  
}Individual;

#endif //__STRUCTS_H

