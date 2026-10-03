#include "cga.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "consts.h"
#include "ca.h"
#include "utils.h"
#include "params.h"
#include "templates.h"
#include "backend.h"

void evaluatePopulation(Individual *pop, Lattice *lat, char *rules)
{
  int i,j,fixed;

  for(i=0;i<POPULATION;i++)
  {
    if(params.representation!=REP_BINARY)
      decodeTemplates(&pop[i]);
    memcpy(&lat[i*MAX_LATS],pop[i].lat,sizeof(Lattice)*MAX_LATS);
    memcpy(&rules[i*RULE_SIZE],pop[i].rule,RULE_SIZE);
  }
  //The whole population in a single backend call
  runCA(lat,rules,POPULATION*MAX_LATS,MAX_LATS);
  for(i=0;i<POPULATION;i++)
  {
    pop[i].fitness=0;
    for(j=0;j<MAX_LATS;j++)
      pop[i].fitness += classify(&lat[i*MAX_LATS+j],pop[i].rule,&fixed);
  }
}

void validateRule(const char *rule, int nICs, double *perf, double *perfStrict)
{
  int j,ok=0,okStrict=0,fixed=0;
  Lattice *lat = (Lattice*)malloc(sizeof(Lattice)*nICs);
  if(lat==NULL)
  {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
  createUnbiasedLattices(lat,nICs);
  runCA(lat,rule,nICs,nICs);
  for(j=0;j<nICs;j++)
  {
    if(classify(&lat[j],rule,&fixed))
    {
      ok++;
      okStrict += fixed;
    }
  }
  *perf       = (double)ok/nICs;
  *perfStrict = (double)okStrict/nICs;
  free(lat);
}

void evolve(Individual *pop)
{
  int r,i;
  double totFit = 0;
  char hex[RULE_SIZE/4+1];
  FILE *fp = NULL;
  Lattice *lat = (Lattice*)malloc(sizeof(Lattice)*POPULATION*MAX_LATS);
  char *rules = (char*)malloc(RULE_SIZE*POPULATION);

  if(lat==NULL || rules==NULL)
  {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
#ifdef F_OUTPUT
  if((fp = fopen(F_OUTPUT_FILE,"w+"))==NULL)
    perror("fopen(" F_OUTPUT_FILE "), the trace will not be written");
  else
    fprintf(fp,"Seed:%u\n",params.seed);
#endif
  for(r=0;r<params.generations;r++)
  {
    evaluatePopulation(pop,lat,rules);
    totFit = 0.0;
    if(fp) fprintf(fp,"Run %3d:",r);
    for(i=0;i<POPULATION;i++)
    {
      if(fp) fprintf(fp,"%3d ",pop[i].fitness);
      totFit+=(double)pop[i].fitness;
    }
    totFit/=(double)MAX_LATS;
    bubbleSort(pop);
    bin2hex(hex,pop[POPULATION-1].rule,RULE_SIZE/4,RULE_SIZE);
    hex[RULE_SIZE/4]='\0';
    if(fp)
    {
      fprintf(fp,"(%3d)\nRule:%s\n",pop[POPULATION-1].fitness,hex);
      fflush(fp);
    }
    fprintf(stderr,"Run %3d:[%3d](%.3f%%)\n",r,pop[POPULATION-1].fitness,totFit);
    //The last generation is only ranked: pop[POPULATION-1] is the best individual found
    if(r==params.generations-1) break;
    crossOver(pop);
    mutate(pop,POPULATION-CROSS_AMOUNT);
    for(i=0;i<POPULATION;i++)
      createRandomLattices(&pop[i]);
  }
  if(fp) fclose(fp);
  free(lat);
  free(rules);
}

void crossOver(Individual *pop)
{
  Individual fat1,fat2,son1,son2;
  int f1_idx,f2_idx,s1_idx,s2_idx;
  int rest = (POPULATION-1)-CROSS_AMOUNT;
  int point = 0; //Crossover point
  int i,k;
  int with_tpl = 0; //Elite individuals that have at least one template
#ifdef DEBUG
  FILE *fp = fopen("logs/crossover.log","w+");
  fprintf(fp,"Crossing over...\n");
#endif
  for(i=rest;i<POPULATION;i++)
    if(pop[i].n_tpl>0)
      with_tpl++;
  k=0;
  for(i=0;i<rest;i+=2)
  {
    //Select 2 fathers from the 20 best individuals for crossing. With templates, both
    //need at least one template, otherwise another pair is drawn (if any pair can qualify)
    do
    {
      f1_idx = rest + uniformDeviate(rand()) * (POPULATION - rest);
      f2_idx = rest + uniformDeviate(rand()) * (POPULATION - rest);
    }while(params.representation!=REP_BINARY && with_tpl>0 && (pop[f1_idx].n_tpl==0 || pop[f2_idx].n_tpl==0));

    //Single point crossover with probability p_c, cut point uniform in [1,RULE_SIZE-1].
    //point=0 means no crossover: the sons are copies of the fathers. With p_c=1 no coin is
    //drawn, so the random sequence is the same as an always-crossing GA (cuCga before p_c).
    if(params.representation==REP_BINARY && (params.cross_rate>=1.0 || uniformDeviate(rand()) < params.cross_rate))
      point = 1 + uniformDeviate(rand()) * (RULE_SIZE-1);
    else
      point = 0;

    //Copy fathers to a temp variable
    memcpy(&fat1,&pop[f1_idx],sizeof(Individual));
    memcpy(&fat2,&pop[f2_idx],sizeof(Individual));

    //Sons start as copies of the fathers, so no field is left uninitialized
    memcpy(&son1,&fat1,sizeof(Individual));
    memcpy(&son2,&fat2,sizeof(Individual));

    //Finnaly, cross the genomes
    memcpy(&son1.rule,       &fat2.rule,       point);
    memcpy(&son1.rule[point],&fat1.rule[point],RULE_SIZE-point);
    memcpy(&son2.rule,       &fat1.rule,       point);
    memcpy(&son2.rule[point],&fat2.rule[point],RULE_SIZE-point);

    //With templates, crossover (probability p_c) swaps one template between the sons
    if(params.representation!=REP_BINARY && son1.n_tpl>0 && son2.n_tpl>0 &&
       (params.cross_rate>=1.0 || uniformDeviate(rand()) < params.cross_rate))
      swapTemplates(&son1,&son2);

    //Set the sons index
    s1_idx = k++;
    s2_idx = k++;
    //And put them on the population
    memcpy(&pop[s1_idx],&son1,sizeof(Individual));
    memcpy(&pop[s2_idx],&son2,sizeof(Individual));
#ifdef DEBUG
    fprintf(fp,"Selecting %d(%d) and %d(%d) as fathers.\n",f1_idx,fat1.fitness, f2_idx,fat2.fitness);
    fprintf(fp,"f1:%.*s\nf2:%.*s\n",RULE_SIZE,fat1.rule,RULE_SIZE,fat2.rule);
    fprintf(fp,"Point:%d\n",point);
    fprintf(fp,"s1:%.*s\ns2:%.*s\n",RULE_SIZE,son1.rule,RULE_SIZE,son2.rule);
#endif
  }
#ifdef DEBUG
  fclose(fp);
#endif
}

void mutate(Individual *pop, size_t amount)
{
  int i,j;
  double rnd=0.0;
  for(i=0;i<amount;i++)
  {
    if(params.representation!=REP_BINARY)
    {
      mutateTemplates(&pop[i]);
      continue;
    }
    for(j=0;j<RULE_SIZE;j++)
    {
      rnd = uniformDeviate(rand());
      if(rnd < params.mut_rate)
        pop[i].rule[j]=(pop[i].rule[j]=='0'?'1':'0');
    }
  }
}

