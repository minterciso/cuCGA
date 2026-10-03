#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>

#include "ca.h"
#include "cga.h"
#include "consts.h"
#include "utils.h"
#include "structs.h"
#include "params.h"
#include "templates.h"

void *start_threads(void *individual)
{
  Individual *ind = (Individual*)individual;
  evolve(ind);
  pthread_exit(ind);
}

int main(int argc, char *argv[])
{
  Individual *population = NULL;
  pthread_t threads[1];
  char hex[RULE_SIZE/4+1];
  double perf, perfStrict;
  int i;

  switch(parseParams(argc,argv))
  {
    case 1:  return EXIT_SUCCESS;
    case -1: return EXIT_FAILURE;
  }
  printParams(stderr);
  srand(params.seed);

  //Only evaluate a given rule
  if(params.validate_hex!=NULL)
  {
    char rule[RULE_SIZE];
    if(!parseRule(params.validate_hex,rule))
    {
      fprintf(stderr,"Invalid rule '%s': expected %d hex digits\n",params.validate_hex,RULE_SIZE/4);
      return EXIT_FAILURE;
    }
    validateRule(rule,params.n_ics,&perf,&perfStrict);
    printf("seed=%u rule=%s nics=%d perf=%.4f perf_strict=%.4f\n",params.seed,params.validate_hex,params.n_ics,perf,perfStrict);
    return EXIT_SUCCESS;
  }

  //On the heap: ~15KB per individual (~150KB with VALIDATE)
  if((population=(Individual*)calloc(params.population,sizeof(Individual)))==NULL)
  {
    perror("calloc");
    return EXIT_FAILURE;
  }
  for(i=0;i<params.population;i++)
  {
    createRandomLattices(&population[i]);
#ifdef USE_BEST
    memset(population[i].rule,'0',RULE_SIZE);
    hex2bin(BEST_CGA,population[i].rule,32,RULE_SIZE);
#endif
#ifndef USE_BEST
    if(params.representation==REP_BINARY)
      createRandomRules(&population[i]);
    else
    {
      createRandomTemplates(&population[i]);
      decodeTemplates(&population[i]);
    }
#endif
  }

#ifdef DEBUG
  int j;
  char fname[FNAME_SIZE];
  memset(fname,'\0',FNAME_SIZE);
  for(i=0;i<params.population;i++)
  {
    snprintf(fname,FNAME_SIZE-1,"logs/individual%03d.log",i);
    FILE *dfp = fopen(fname,"w+");
    if(dfp==NULL)
    {
      perror(fname);
      continue;
    }
    fprintf(dfp,"Individual %03d\n",i);
    for(j=0;j<MAX_LATS;j++)
      fprintf(dfp,"Lat %03d(%3d):%.*s\n",j,population[i].lat[j].density,LAT_SIZE,population[i].lat[j].cells);
    fprintf(dfp,"Rule: %.*s\n",RULE_SIZE,population[i].rule);
    fclose(dfp);
  }
#endif

  //Now we run the threaded part
  for(i=0;i<1;i++)
  {
    population[i].id = i; //Set before the thread starts reading the population
    pthread_create(&threads[i],NULL,&start_threads,population);
  }
  for(i=0;i<1;i++)
    pthread_join(threads[i],NULL);

  //The population is ranked at the last generation and not altered afterwards,
  //so the last individual is the best one found
  char rule_dec[RULE_DEC_SIZE];
  Individual *best = &population[params.population-1];
  ruleToDecimal(best->rule,rule_dec);
  fprintf(stderr,"Best rule: %s (fitness %u)\n",rule_dec,best->fitness);
  if(params.representation!=REP_BINARY)
  {
    fprintf(stderr,"Templates (orientation %c):",best->orientation);
    for(i=0;i<best->n_tpl;i++)
      fprintf(stderr," %.*s",NEIGH_SIZE,best->tpl[i].cells);
    fprintf(stderr,"\n");
  }

  //Final evaluation of the best rule on binomially distributed ICs
  bin2hex(hex,best->rule,RULE_SIZE/4,RULE_SIZE);
  hex[RULE_SIZE/4]='\0';
  validateRule(best->rule,params.n_ics,&perf,&perfStrict);
#ifdef F_OUTPUT
  FILE *fp = fopen(F_OUTPUT_FILE,"a");
  if(fp!=NULL)
  {
    fprintf(fp,"Performance:%.4f Strict:%.4f ICs:%d\n",perf,perfStrict,params.n_ics);
    fclose(fp);
  }
#endif
  printf("seed=%u train_best=%u rule=%s nics=%d perf=%.4f perf_strict=%.4f\n",
         params.seed,best->fitness,hex,params.n_ics,perf,perfStrict);

  free(population);
  return EXIT_SUCCESS;
}
