#include "cga.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "consts.h"
#include "ca.h"
#include "utils.h"
#include "params.h"
#include "templates.h"
#include "backend.h"

void evaluatePopulation(Individual *pop, const Lattice *ics, Lattice *lat, char *rules)
{
  int i,j,fixed;
  int N = params.n_train_ics;

  for(i=0;i<params.population;i++)
  {
    if(params.representation!=REP_BINARY)
      decodeTemplates(&pop[i]);
    //The CA runs in place, so shared ICs are copied once per individual
    memcpy(&lat[(size_t)i*N],params.shared_ics ? ics : &ics[(size_t)i*N],sizeof(Lattice)*N);
    memcpy(&rules[(size_t)i*RULE_SIZE],pop[i].rule,RULE_SIZE);
  }
  //The whole population in a single backend call
  runCA(lat,rules,params.population*N,N);
  for(i=0;i<params.population;i++)
  {
    pop[i].fitness=0;
    for(j=0;j<N;j++)
      pop[i].fitness += classify(&lat[(size_t)i*N+j],pop[i].rule,&fixed);
  }
}

void validateRule(const char *rule, int nICs, double *perf, double *perfStrict)
{
  validateRules(rule,1,nICs,perf,perfStrict);
}

//Lattices per backend call when validating many rules (~300MB of Lattice)
#define VALIDATE_CHUNK_LATS (1<<21)

void validateRules(const char *rules, int nRules, int nICs, double *perf, double *perfStrict)
{
  int r0,k,j,c,ok,okStrict,fixed=0;
  int chunk = nICs >= VALIDATE_CHUNK_LATS ? 1 : VALIDATE_CHUNK_LATS/nICs;
  if(chunk > nRules) chunk = nRules;
  Lattice *ics = (Lattice*)malloc(sizeof(Lattice)*nICs);
  Lattice *lat = (Lattice*)malloc(sizeof(Lattice)*nICs*(size_t)chunk);
  if(ics==NULL || lat==NULL)
  {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
  //One IC set for every rule; the CA runs in place, so each rule gets a copy
  createUnbiasedLattices(ics,nICs);
  for(r0=0;r0<nRules;r0+=chunk)
  {
    c = (nRules-r0 < chunk) ? nRules-r0 : chunk;
    for(k=0;k<c;k++)
      memcpy(&lat[(size_t)k*nICs],ics,sizeof(Lattice)*nICs);
    runCA(lat,&rules[(size_t)r0*RULE_SIZE],c*nICs,nICs);
    for(k=0;k<c;k++)
    {
      ok = okStrict = 0;
      for(j=0;j<nICs;j++)
      {
        if(classify(&lat[(size_t)k*nICs+j],&rules[(size_t)(r0+k)*RULE_SIZE],&fixed))
        {
          ok++;
          okStrict += fixed;
        }
      }
      perf[r0+k]       = (double)ok/nICs;
      perfStrict[r0+k] = (double)okStrict/nICs;
    }
  }
  free(ics);
  free(lat);
}

void evolve(Individual *pop, Lattice *ics)
{
  int r,i;
  double totFit = 0;
  char hex[RULE_SIZE/4+1];
  FILE *fp = NULL;
  FILE *csv = NULL;
  double sumSq, eliteSum, mean;
  int P = params.population;
  int N = params.n_train_ics;
  Lattice *lat = (Lattice*)malloc(sizeof(Lattice)*N*(size_t)P);
  char *rules = (char*)malloc((size_t)RULE_SIZE*P);

  if(lat==NULL || rules==NULL)
  {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
  if(params.csv_path!=NULL)
  {
    if((csv = fopen(params.csv_path,"w"))==NULL)
    {
      perror(params.csv_path);
      exit(EXIT_FAILURE);
    }
    fprintf(csv,"generation,best,elite_mean,mean,std,min,best_rule\n");
  }
#ifdef F_OUTPUT
  if((fp = fopen(F_OUTPUT_FILE,"w+"))==NULL)
    perror("fopen(" F_OUTPUT_FILE "), the trace will not be written");
  else
    fprintf(fp,"Seed:%u\n",params.seed);
#endif
  for(r=0;r<params.generations;r++)
  {
    evaluatePopulation(pop,ics,lat,rules);
    totFit = 0.0;
    if(fp) fprintf(fp,"Run %3d:",r);
    for(i=0;i<P;i++)
    {
      if(fp) fprintf(fp,"%3d ",pop[i].fitness);
      totFit+=(double)pop[i].fitness;
    }
    totFit=100.0*totFit/((double)P*N); //Average fitness, in % of the ICs
    sortByFitness(pop,P,N);
    bin2hex(hex,pop[P-1].rule,RULE_SIZE/4,RULE_SIZE);
    hex[RULE_SIZE/4]='\0';
    if(fp)
    {
      fprintf(fp,"(%3d)\nRule:%s\n",pop[P-1].fitness,hex);
      fflush(fp);
    }
    fprintf(stderr,"Run %3d:[%3d](%.3f%%)\n",r,pop[P-1].fitness,totFit);
    if(csv)
    {
      //Sorted ascending: pop[0] is the worst, pop[P-1] the best, the elite are the last ones
      sumSq = eliteSum = mean = 0.0;
      for(i=0;i<P;i++)
        mean += pop[i].fitness;
      mean /= P;
      for(i=0;i<P;i++)
        sumSq += (pop[i].fitness-mean)*(pop[i].fitness-mean);
      for(i=P-params.elite;i<P;i++)
        eliteSum += pop[i].fitness;
      fprintf(csv,"%d,%u,%.4f,%.4f,%.4f,%u,%s\n",r,pop[P-1].fitness,eliteSum/params.elite,mean,
              sqrt(sumSq/P),pop[0].fitness,hex);
      fflush(csv);
    }
    //The last generation is only ranked: pop[P-1] is the best individual found
    if(r==params.generations-1) break;
    crossOver(pop);
    mutate(pop,P-params.elite);
    if(params.shared_ics)
      createTrainingLattices(ics,N);
    else
      for(i=0;i<P;i++)
        createTrainingLattices(&ics[(size_t)i*N],N);
  }
  if(fp) fclose(fp);
  if(csv) fclose(csv);
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

