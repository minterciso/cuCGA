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

typedef struct EvolveArgs
{
  Individual *pop;
  Lattice *ics;
}EvolveArgs;

void *start_threads(void *arg)
{
  EvolveArgs *ea = (EvolveArgs*)arg;
  evolve(ea->pop,ea->ics);
  pthread_exit(ea);
}

//-V: one hex rule per line (blank lines and #-comments skipped); CSV to stdout
static int validateFile(const char *path)
{
  FILE *fp = fopen(path,"r");
  char line[256];
  char *rules = NULL, *hexes = NULL;
  double *perf, *perfStrict;
  int n = 0, cap = 0, i, lineNo = 0;
  size_t len;

  if(fp==NULL)
  {
    perror(path);
    return EXIT_FAILURE;
  }
  while(fgets(line,sizeof(line),fp)!=NULL)
  {
    lineNo++;
    len = strcspn(line," \t\r\n");
    line[len] = '\0';
    if(len==0 || line[0]=='#')
      continue;
    if(n==cap)
    {
      cap = cap ? 2*cap : 1024;
      rules = (char*)realloc(rules,(size_t)cap*RULE_SIZE);
      hexes = (char*)realloc(hexes,(size_t)cap*(RULE_SIZE/4+1));
      if(rules==NULL || hexes==NULL)
      {
        perror("realloc");
        return EXIT_FAILURE;
      }
    }
    if(len!=RULE_SIZE/4 || !parseRule(line,&rules[(size_t)n*RULE_SIZE]))
    {
      fprintf(stderr,"%s:%d: invalid rule '%s': expected %d hex digits\n",path,lineNo,line,RULE_SIZE/4);
      return EXIT_FAILURE;
    }
    memcpy(&hexes[(size_t)n*(RULE_SIZE/4+1)],line,RULE_SIZE/4+1);
    n++;
  }
  fclose(fp);
  if(n==0)
  {
    fprintf(stderr,"%s: no rules\n",path);
    return EXIT_FAILURE;
  }
  perf = (double*)malloc(sizeof(double)*n);
  perfStrict = (double*)malloc(sizeof(double)*n);
  if(perf==NULL || perfStrict==NULL)
  {
    perror("malloc");
    return EXIT_FAILURE;
  }
  validateRules(rules,n,params.n_ics,perf,perfStrict);
  printf("rule,nics,seed,perf,perf_strict\n");
  for(i=0;i<n;i++)
    printf("%s,%d,%u,%.6f,%.6f\n",&hexes[(size_t)i*(RULE_SIZE/4+1)],params.n_ics,params.seed,perf[i],perfStrict[i]);
  free(rules);
  free(hexes);
  free(perf);
  free(perfStrict);
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
  Individual *population = NULL;
  Lattice *ics = NULL;
  EvolveArgs ea;
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

  //Only evaluate the rules of a file, all on the same ICs
  if(params.validate_file!=NULL)
    return validateFile(params.validate_file);

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

  //Training ICs: one set for the whole population, or one per individual (see evolve())
  int N = params.n_train_ics;
  size_t nSets = params.shared_ics ? 1 : (size_t)params.population;
  if((population=(Individual*)calloc(params.population,sizeof(Individual)))==NULL ||
     (ics=(Lattice*)malloc(sizeof(Lattice)*N*nSets))==NULL)
  {
    perror("calloc");
    return EXIT_FAILURE;
  }
  if(params.shared_ics)
    createTrainingLattices(ics,N);
  for(i=0;i<params.population;i++)
  {
    //Drawn interleaved with the rules, so per-individual runs keep the random sequence of earlier versions
    if(!params.shared_ics)
      createTrainingLattices(&ics[(size_t)i*N],N);
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
    Lattice *ind_ics = params.shared_ics ? ics : &ics[(size_t)i*N];
    for(j=0;j<N;j++)
      fprintf(dfp,"Lat %03d(%3d):%.*s\n",j,ind_ics[j].density,LAT_SIZE,ind_ics[j].cells);
    fprintf(dfp,"Rule: %.*s\n",RULE_SIZE,population[i].rule);
    fclose(dfp);
  }
#endif

  //Now we run the threaded part
  ea.pop = population;
  ea.ics = ics;
  for(i=0;i<1;i++)
  {
    population[i].id = i; //Set before the thread starts reading the population
    pthread_create(&threads[i],NULL,&start_threads,&ea);
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
  free(ics);
  return EXIT_SUCCESS;
}
