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

  for(i=0;i<params.population;i++)
  {
    if(params.representation!=REP_BINARY)
      decodeTemplates(&pop[i]);
    memcpy(&lat[(size_t)i*MAX_LATS],pop[i].lat,sizeof(Lattice)*MAX_LATS);
    memcpy(&rules[(size_t)i*RULE_SIZE],pop[i].rule,RULE_SIZE);
  }
  //The whole population in a single backend call
  runCA(lat,rules,params.population*MAX_LATS,MAX_LATS);
  for(i=0;i<params.population;i++)
  {
    pop[i].fitness=0;
    for(j=0;j<MAX_LATS;j++)
      pop[i].fitness += classify(&lat[(size_t)i*MAX_LATS+j],pop[i].rule,&fixed);
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
  int P = params.population;
  Lattice *lat = (Lattice*)malloc(sizeof(Lattice)*MAX_LATS*(size_t)P);
  char *rules = (char*)malloc((size_t)RULE_SIZE*P);

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
    for(i=0;i<P;i++)
    {
      if(fp) fprintf(fp,"%3d ",pop[i].fitness);
      totFit+=(double)pop[i].fitness;
    }
    totFit=100.0*totFit/((double)P*MAX_LATS); //Average fitness, in % of the ICs
    bubbleSort(pop,P);
    bin2hex(hex,pop[P-1].rule,RULE_SIZE/4,RULE_SIZE);
    hex[RULE_SIZE/4]='\0';
    if(fp)
    {
      fprintf(fp,"(%3d)\nRule:%s\n",pop[P-1].fitness,hex);
      fflush(fp);
    }
    fprintf(stderr,"Run %3d:[%3d](%.3f%%)\n",r,pop[P-1].fitness,totFit);
    //The last generation is only ranked: pop[P-1] is the best individual found
    if(r==params.generations-1) break;
    crossOver(pop);
    mutate(pop,P-params.elite);
    for(i=0;i<P;i++)
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
  int P = params.population;
  int n_sons = P-params.elite; //Non-elite slots, replaced by the sons
  //Parent pool: indices rest..P-1 after ranking, i.e. the elite plus the next best individual
  //(elite+1 individuals, as in the original code)
  int rest = (P-1)-params.elite;
  int point = 0; //Crossover point
  int i,k;
  int with_tpl = 0; //Elite individuals that have at least one template
#ifdef DEBUG
  FILE *fp = fopen("logs/crossover.log","w+");
  fprintf(fp,"Crossing over...\n");
#endif
  for(i=rest;i<P;i++)
    if(pop[i].n_tpl>0)
      with_tpl++;
  k=0;
  while(k<n_sons)
  {
    //Select 2 fathers from the best individuals for crossing. With templates, both
    //need at least one template, otherwise another pair is drawn (if any pair can qualify)
    do
    {
      f1_idx = rest + uniformDeviate(rand()) * (P - rest);
      f2_idx = rest + uniformDeviate(rand()) * (P - rest);
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

    //Set the sons index and put them on the population; with an odd number of non-elite
    //slots the last pair only places its first son
    s1_idx = k++;
    memcpy(&pop[s1_idx],&son1,sizeof(Individual));
    if(k<n_sons)
    {
      s2_idx = k++;
      memcpy(&pop[s2_idx],&son2,sizeof(Individual));
    }
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

