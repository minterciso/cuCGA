#include "ca.h"
#include "utils.h"

#include <assert.h>

void createRandomLattice(Individual *ind)
{
  assert(ind!=NULL);
  int i;
  int count = 0;
  int rnd   = 0;

  for(i=0;i<MAX_LATS;i++)
  {
    memset(ind->lat[i].cells,'0',LAT_SIZE); //We allways start with an empty Lattice
#ifndef VALIDATE
    ind->lat[i].density = uniformDeviate(rand())*(LAT_SIZE-1); //Uniform distribution
#endif
#ifdef VALIDATE
    ind->lat[i].density = 90 + uniformDeviate(rand())*(60-90); //Unbiased distribution
#endif
    count = 0;
    while(count <= ind->lat[i].density)
    {
      rnd = uniformDeviate(rand())*(LAT_SIZE-1);  //All cells have equal probability to be choosen
      if(ind->lat[i].cells[rnd]=='0')
      {
        ind->lat[i].cells[rnd]='1';
        count++;
      }
    }
  }
}

void createRandomRules(Individual *ind)
{
  int i;
  int rnd = 0;
  memset(ind->rule,'0',RULE_SIZE);
  for(i=0;i<RULE_SIZE;i++)
  {
    rnd = uniformDeviate(rand())*MODE;
    switch(rnd)
    {
      case 0:ind->rule[i]='0';break;
      case 1:ind->rule[i]='1';break;
      case 2:ind->rule[i]='2';break;
    }
  }
}

