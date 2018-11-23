#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "utils.h"
#include "structs.h"
#include "consts.h"
#include "ca.h"
#include "cga.h"

int main(int argc, char *argv[])
{
  Individual *h_pop;
  Lattice *h_lat,*h_blat;
  size_t popSize = sizeof(Individual)*POPULATION;
  size_t latSize = sizeof(Lattice)*MAX_LATS;
  size_t blatSize= sizeof(Lattice)*MAX_BIN_LATS;
  int i,j;

  if((h_pop=(Individual*)malloc(popSize))==NULL)
  {
    perror("malloc");
    return EXIT_FAILURE;
  }
  if((h_lat=(Lattice*)malloc(latSize))==NULL)
  {
    perror("malloc");
    free(h_pop);
    return EXIT_FAILURE;
  }
  if((h_blat=(Lattice*)malloc(blatSize))==NULL)
  {
    perror("malloc");
    free(h_pop);
    free(h_blat);
    return EXIT_FAILURE;
  }

  FILE *out = fopen("logs/result.csv","w+");
  if(out==NULL)
    out=stderr;

  fprintf(out,"Execution,Fitness\n");
  fflush(out);
  //Initial values
  srand(timeSeed());
  for(int r=0;r<MAX_RUNS;r++)
  {
    memset(h_pop,'\0',popSize);
    memset(h_lat,'\0',latSize);

    for(i=0;i<MAX_LATS;i++)
    {
      h_lat[i].density = uniformDeviate(rand())*(LAT_SIZE-1);
      createRandomLattice(h_lat[i].cells,h_lat[i].density,0);
    }
    createRandomRules(h_pop);

#ifdef USE_BEST
    char best[33];
    snprintf(best,33,"0504058605000f77037755877bffb77f\0");
    for(i=0;i<POPULATION;i++)
    {
      h_pop[i].fitness = 0;
      memset(h_pop[i].rule,'0',RULE_SIZE);
      hex2bin(best,h_pop[i].rule,32,RULE_SIZE);
      h_pop[i].rule[RULE_SIZE]='\0';
    }
#endif
    if(cga(h_pop,h_lat)!=0)
    {
      fprintf(stderr,"Error running CGA\n");
      if(h_pop  != NULL) free(h_pop);
      if(h_lat  != NULL) free(h_lat);
      if(h_blat != NULL) free(h_blat);
      return EXIT_FAILURE;
    }
    h_pop[POPULATION-1].fitness=0;
    for(i=0;i<MAX_BIN_LATS;i++)
    {
      h_blat[i].density = createRandomLattice(h_blat[i].cells,0,1);
      fprintf(stdout,"Calculating Fitness using 10.000 binomial random IC:%05d\t%3.2f%%",h_pop[POPULATION-1].fitness,((float)(i+1)/(float)MAX_BIN_LATS)*100);
      testCga(&h_pop[POPULATION-1],&h_blat[i]);
      fprintf(stdout,"\r");
    }
    fprintf(stdout,"\nBest Fitness:%f\n",(float)((float)h_pop[POPULATION-1].fitness/(float)MAX_BIN_LATS));
    fprintf(out,"%d,%f\n",r,(float)((float)h_pop[POPULATION-1].fitness/(float)MAX_BIN_LATS));
    fflush(out);
  }
    

  if(h_pop  != NULL) free(h_pop);
  if(h_lat  != NULL) free(h_lat);
  if(h_blat != NULL) free(h_blat);
  if(out!=stderr) fclose(out);

  return EXIT_SUCCESS;
}
