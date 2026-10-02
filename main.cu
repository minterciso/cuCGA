#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>

#include "structs.h"
#include "consts.h"
#include "ca.h"
#include "cga.h"
#include "utils.h"
#include "kernel.h"

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t _err = (call);                                                     \
    if(_err != cudaSuccess)                                                        \
    {                                                                              \
      fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_err)); \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  } while(0)

//Run nLats lattices on the device; lattice j uses rule j/latsPerRule from rules
static void runCA(Lattice *h_lat, const char *h_rules, int nLats, int latsPerRule)
{
  Lattice *d_lat;
  char *d_rules;
  size_t latSize  = sizeof(Lattice)*nLats;
  size_t ruleSize = (size_t)RULE_SIZE*((nLats+latsPerRule-1)/latsPerRule);

  CUDA_CHECK(cudaMalloc((void**)&d_lat,latSize));
  CUDA_CHECK(cudaMalloc((void**)&d_rules,ruleSize));
  CUDA_CHECK(cudaMemcpy(d_lat,h_lat,latSize,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rules,h_rules,ruleSize,cudaMemcpyHostToDevice));

  dim3 blockSize(128);
  dim3 gridSize((nLats+blockSize.x-1)/blockSize.x);
  executeCA<<<gridSize,blockSize>>>(d_lat,d_rules,nLats,latsPerRule);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_lat,d_lat,latSize,cudaMemcpyDeviceToHost));
  cudaFree(d_lat);
  cudaFree(d_rules);
}

//Returns 0 for wrong, 1 for correct (final lattice uniform with the majority bit), and sets
//*fixed to whether that uniform state is also a fixed point of the rule
static int classify(const Lattice *lat, const char *rule, int *fixed)
{
  int count = 0;
  for(int k=0;k<LAT_SIZE;k++)
    if(lat->cells[k]=='1') count++;
  if(lat->density > LAT_SIZE/2 && count==LAT_SIZE)
  {
    *fixed = (rule[RULE_SIZE-1]=='1');
    return 1;
  }
  if(lat->density <= LAT_SIZE/2 && count==0)
  {
    *fixed = (rule[0]=='0');
    return 1;
  }
  *fixed = 0;
  return 0;
}

//Performance on nICs unbiased (binomial density) ICs, as in the final evaluation of MCH/CMD.
//perf counts a uniform correct final state after CA_RUNS steps (same criterion as the GA fitness);
//perfStrict additionally requires that state to be a fixed point of the rule.
static void validateRule(const char *rule, int nICs, double *perf, double *perfStrict)
{
  Lattice *lat = (Lattice*)malloc(sizeof(Lattice)*nICs);
  if(lat==NULL)
  {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
  createUnbiasedLattices(lat,nICs);
  runCA(lat,rule,nICs,nICs);
  int ok = 0, okStrict = 0, fixed = 0;
  for(int j=0;j<nICs;j++)
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

static int parseRule(const char *hex, char *rule)
{
  if(strlen(hex)!=RULE_SIZE/4 || strspn(hex,"0123456789abcdefABCDEF")!=RULE_SIZE/4)
    return 0;
  char buf[RULE_SIZE/4+1];
  strcpy(buf,hex);
  memset(rule,'0',RULE_SIZE);
  hex2bin(buf,rule,RULE_SIZE/4,RULE_SIZE);
  return 1;
}

static void usage(const char *prog)
{
  fprintf(stderr,"Usage: %s [-s seed] [-n validation_ics] [-v rule_hex]\n",prog);
  fprintf(stderr,"  -s seed   RNG seed (default: derived from time)\n");
  fprintf(stderr,"  -n N      number of binomial ICs for the final evaluation (default: 10000)\n");
  fprintf(stderr,"  -v HEX    only evaluate the given 32-digit hex rule on N binomial ICs, no GA\n");
}

int main(int argc, char *argv[])
{
  Individual *h_pop;
  Lattice *h_lat;
  char *h_rules;
  size_t h_memSize = sizeof(Individual)*POPULATION;
  unsigned int seed = timeSeed();
  int nICs = 10000;
  const char *validateHex = NULL;
  int opt;

  while((opt = getopt(argc,argv,"s:n:v:h")) != -1)
  {
    switch(opt)
    {
      case 's': seed = (unsigned int)strtoul(optarg,NULL,10); break;
      case 'n': nICs = atoi(optarg); break;
      case 'v': validateHex = optarg; break;
      default:
        usage(argv[0]);
        return EXIT_FAILURE;
    }
  }
  if(nICs <= 0)
  {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  srand(seed);

  if(validateHex != NULL)
  {
    char rule[RULE_SIZE];
    double perf, perfStrict;
    if(!parseRule(validateHex,rule))
    {
      fprintf(stderr,"Invalid rule '%s': expected %d hex digits\n",validateHex,RULE_SIZE/4);
      return EXIT_FAILURE;
    }
    validateRule(rule,nICs,&perf,&perfStrict);
    printf("seed=%u rule=%s nics=%d perf=%.4f perf_strict=%.4f\n",seed,validateHex,nICs,perf,perfStrict);
    return EXIT_SUCCESS;
  }

  //Allocate host memory
  h_pop   = (Individual*)malloc(h_memSize);
  h_lat   = (Lattice*)malloc(sizeof(Lattice)*POPULATION*MAX_LATS);
  h_rules = (char*)malloc(RULE_SIZE*POPULATION);
  if(h_pop==NULL || h_lat==NULL || h_rules==NULL)
  {
    perror("malloc");
    return EXIT_FAILURE;
  }
  memset(h_pop,'\0',h_memSize);

  for(int i = 0; i < POPULATION; i++)
  {
    createRandomLattice(&h_pop[i]);
#ifndef USE_BEST
    createRandomRules(&h_pop[i]);
#endif
#ifdef USE_BEST
    memset(h_pop[i].rule,'0',RULE_SIZE);
    hex2bin((char*)BEST_CGA,h_pop[i].rule,32,RULE_SIZE);
#endif
  }

  FILE *fp = fopen(F_OUTPUT_FILE,"w+");
  if(fp==NULL)
  {
    perror("fopen(" F_OUTPUT_FILE ")");
    return EXIT_FAILURE;
  }
  fprintf(fp,"Seed:%u\n",seed);

  char hex[33];
  for(int g=0;g<GA_RUNS;g++)
  {
    fprintf(stderr,".");
    fprintf(fp,"Run %3d:",g);
    //Evaluate the whole population in a single kernel launch
    for(int i = 0; i < POPULATION;i++)
    {
      memcpy(&h_lat[i*MAX_LATS],h_pop[i].lat,sizeof(Lattice)*MAX_LATS);
      memcpy(&h_rules[i*RULE_SIZE],h_pop[i].rule,RULE_SIZE);
    }
    runCA(h_lat,h_rules,POPULATION*MAX_LATS,MAX_LATS);
    for(int i = 0; i < POPULATION;i++)
    {
      int fixed;
      memcpy(h_pop[i].lat,&h_lat[i*MAX_LATS],sizeof(Lattice)*MAX_LATS);
      for(int j = 0;j <MAX_LATS;j++)
        h_pop[i].fitness += classify(&h_pop[i].lat[j],h_pop[i].rule,&fixed);
      fprintf(fp,"%3d ",h_pop[i].fitness);
    }
    bubbleSort(h_pop);
    fprintf(fp,"(%3d)\n",h_pop[POPULATION-1].fitness);
    memset(hex,'0',32);
    bin2hex(hex,h_pop[POPULATION-1].rule,32,RULE_SIZE);
    hex[32]='\0';
    fprintf(fp,"Rule:%s\n",hex);
    fflush(fp);
    if(g==GA_RUNS-1) break;
    crossOver(h_pop);
    mutate(h_pop,POPULATION-CROSS_AMOUNT);
    for(int i = 0; i < POPULATION; i++)
    {
      createRandomLattice(&h_pop[i]);
      h_pop[i].fitness=0;
    }
  }
  fprintf(stderr,"\n");

  //Final evaluation of the best rule on binomially distributed ICs
  double perf, perfStrict;
  validateRule(h_pop[POPULATION-1].rule,nICs,&perf,&perfStrict);
  fprintf(fp,"Performance:%.4f Strict:%.4f ICs:%d\n",perf,perfStrict,nICs);
  printf("seed=%u train_best=%d rule=%s nics=%d perf=%.4f perf_strict=%.4f\n",
         seed,h_pop[POPULATION-1].fitness,hex,nICs,perf,perfStrict);

  fclose(fp);
  //Clear memory
  free(h_pop);
  free(h_lat);
  free(h_rules);

  return EXIT_SUCCESS;
}
