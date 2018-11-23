#include "ca.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>

void createRandomDens(int *dens)
{
  int i;
  for(i=0;i<MAX_LATS;i++)
    dens[i] = uniformDeviate(rand())*(LAT_SIZE-1);
}

int createRandomLattice(char *lattice,int dens,int binomial)
{
  memset(lattice,'0',LAT_SIZE);
  lattice[LAT_SIZE]='\0';
  if(binomial==0)
  {
    int count,idx;
    count=0;
    while(count < dens)
    {
      idx=uniformDeviate(rand())*(LAT_SIZE-1);
      if(lattice[idx]=='0')
      {
        lattice[idx]='1';
        count++;
      }
    }
  }
  else
  {
    int i;
    int d=0;
    for(i=0;i<LAT_SIZE;i++)
    {
      lattice[i]='0' + uniformDeviate(rand())*2;
      if(lattice[i]=='1') d++;
    }
    return d;
  }
  return 0;
}

void createRandomRules(Individual *population)
{
  int i,j;
  for(i=0;i<POPULATION;i++)
  {
    for(j=0;j<RULE_SIZE;j++)
      population[i].rule[j] = '0' + uniformDeviate(rand())*2;
    population[i].rule[RULE_SIZE]='\0';
    population[i].fitness=0;
  }
}

void testCga(Individual *ind, Lattice *lat)
{
  int pos,dif;
  int i,j,run;
  int finalDens;
  char bin[RADIUS*2+1];
  char next[LAT_SIZE];

  for(run=0;run<CA_RUNS;run++)
  {
    memset(next,'0',LAT_SIZE);

    for(i=0;i<LAT_SIZE;i++)
    {
      dif = i-RADIUS;
      if(dif < 0) pos = LAT_SIZE+dif;
      else pos = dif;

      for(j=0;j<RADIUS*2+1;j++)
      {
        bin[j] = lat->cells[pos++];
        if(pos==LAT_SIZE)
          pos = 0;
      }
      next[i] = ind->rule[bin2dec(bin)];
    }
    
    for(j=0;j<LAT_SIZE;j++){ if(lat->cells[j]!=next[j]) break;}
    if(j==LAT_SIZE) break;


    for(i=0;i<LAT_SIZE;i++)
      lat->cells[i]=next[i];
  }

  finalDens=0;
  for(i=0;i<LAT_SIZE;i++)
  {
    if(lat->cells[i]=='1') 
      finalDens++;
  }

  if( (lat->density > LAT_SIZE/2 && finalDens == LAT_SIZE) ||
      (lat->density < LAT_SIZE/2 && finalDens == 0)          )
    ind->fitness++; 
}
