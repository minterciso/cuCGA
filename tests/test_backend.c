#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"
#include "ca.h"
#include "utils.h"

//The GPU backend (runCA) must give exactly the same final lattices as the CPU reference
//(cpuRunCA) for random rules on random and unbiased ICs, first with CA_RUNS steps and then
//with a different step count per lattice (both parities, beyond CA_RUNS too, and 0).
//N_RULES*LATS_PER_RULE is odd, so the last block of the kernel is only partly used.
#define N_RULES 7
#define LATS_PER_RULE 501

int main(void)
{
  int nLats = N_RULES*LATS_PER_RULE;
  Lattice *gpu = (Lattice*)malloc(sizeof(Lattice)*nLats);
  Lattice *cpu = (Lattice*)malloc(sizeof(Lattice)*nLats);
  char rules[N_RULES*RULE_SIZE];
  int i,k,pass,diff=0;

  if(gpu==NULL || cpu==NULL)
  {
    perror("malloc");
    return 1;
  }
  srand(2026);
  //Rule 0 is GKL (converges), rule 1 all zeros (fixed point after one step), rule 2 the
  //complement of the centre cell (period 2, never fixed: runs all CA_RUNS steps), the
  //others random
  parseRule("005f005f005f005f005fff5f005fff5f",&rules[0]);
  memset(&rules[RULE_SIZE],'0',RULE_SIZE);
  for(k=0;k<RULE_SIZE;k++)
    rules[2*RULE_SIZE+k] = ((k>>RADIUS)&1) ? '0' : '1';
  for(i=3*RULE_SIZE;i<N_RULES*RULE_SIZE;i++)
    rules[i] = (uniformDeviate(rand()) < 0.5 ? '0' : '1');
  createUnbiasedLattices(gpu,nLats);
  //Make half of them low/high density so convergence is exercised too
  for(i=0;i<nLats;i+=2)
    for(k=0;k<LAT_SIZE;k++)
      if(gpu[i].cells[k]=='1' && uniformDeviate(rand()) < 0.5)
        gpu[i].cells[k]='0';
  for(pass=0;pass<2;pass++)
  {
    int d=0;
    if(pass==1)
      for(i=0;i<nLats;i++)
        gpu[i].steps = (i%37==0) ? 0 : poissonDeviate(POISSON_STEPS_MEAN) + (i&1);
    memcpy(cpu,gpu,sizeof(Lattice)*nLats);

    runCA(gpu,rules,nLats,LATS_PER_RULE);
    cpuRunCA(cpu,rules,nLats,LATS_PER_RULE);

    for(i=0;i<nLats;i++)
      if(memcmp(gpu[i].cells,cpu[i].cells,LAT_SIZE)!=0)
        d++;
    if(d)
      fprintf(stderr,"FAIL: %d of %d lattices differ between GPU and CPU (%s steps)\n",d,nLats,pass ? "random" : "fixed");
    else
      printf("GPU and CPU agree on all %d lattices (%s steps)\n",nLats,pass ? "random" : "fixed");
    diff += d;
    //Fresh unbiased ICs for the second pass
    if(pass==0)
    {
      srand(77);
      createUnbiasedLattices(gpu,nLats);
    }
  }
  free(gpu);
  free(cpu);
  return diff==0 ? 0 : 1;
}
