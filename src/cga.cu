#include "cga.h"
#include "consts.h"
#include "utils.h"

#include <cuda.h>
#include <errno.h>
#include <string.h>

__device__ int bin2dec(char *binary);
__device__ int executeCA(Individual* ind, Lattice *lat);
__global__ void cgaKernel(Individual* d_pop, Lattice* d_lat);

int __printErr(FILE *stream, cudaError_t error, char *string)
{
  if(error != cudaSuccess)
  {
    if(stream==NULL)
      stream = stderr;

    if(string!=NULL) fprintf(stream,"[ERROR] %s: %s\n",string,cudaGetErrorString(error));
    else fprintf(stream,"[ERROR] : %s\n",cudaGetErrorString(error));
    return 1;
  }
  return 0;
}

int cga(Individual *h_pop, Lattice* h_lat)
{
  size_t popSize = sizeof(Individual)*POPULATION;
  size_t latSize = sizeof(Lattice)*MAX_LATS;
  cudaError_t d_error;
  Individual *d_pop;
  Lattice *d_lat;

  
  //Allocate memory
  fprintf(stdout,"[*] Allocating and copying GPU memory...\n");
  d_error = cudaMalloc((void**)&d_pop,popSize);
  if(__printErr(stderr,d_error,"cudaMalloc()")!=0) return 1;
  d_error = cudaMalloc((void**)&d_lat,latSize);
  if(__printErr(stderr,d_error,"cudaMalloc()")!=0) return 1; 

  d_error = cudaMemcpy(d_lat,h_lat,latSize,cudaMemcpyHostToDevice);
  if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;

  //Re-setting the fitness
  //We evolve GA_RUNS time
  fprintf(stdout,"[*] Starting GPU process...\n");
  for(int g = 0; g < GA_RUNS; g++)
  {
    for(int i = 0; i < POPULATION;i++)
      h_pop[i].fitness = 0;

    //Pass to device
    d_error = cudaMemcpy(d_pop,h_pop,popSize,cudaMemcpyHostToDevice);
    if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;

    fprintf(stderr,"Evolving\t%2.2f%%",(float)((float)(g+1)/(float)GA_RUNS)*100);

    //Call the main Kernel
    dim3 grid(MAX_LATS);
    dim3 blocks(POPULATION);
    cgaKernel<<<grid,blocks>>>(d_pop,d_lat);
    d_error = cudaThreadSynchronize();
    if(__printErr(stderr,d_error,"Kernel")!=0) return 1;

    //Copy back
    d_error = cudaMemcpy(h_pop,d_pop,popSize,cudaMemcpyDeviceToHost);
    if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;

    //Set the correct NULL
    for(int i=0;i<POPULATION;i++)
      h_pop[i].rule[RULE_SIZE]='\0';

    bubbleSort(h_pop);
    /*
    fprintf(stderr,"%03d:\n",g);
    for(int i=0;i<POPULATION;i++)
      fprintf(stderr,"[%03d] ",h_pop[i].fitness);
    fprintf(stderr,"\n");
    */
    fprintf(stderr,"\r");
    if(g==GA_RUNS-1) break;
    crossOver(h_pop);
    mutate(h_pop);
  }

  //Clean up
  fprintf(stderr,"[*] Cleaning up...\n");
  if(d_pop!=NULL) cudaFree(d_pop);
  if(d_lat!=NULL) cudaFree(d_lat);

  return 0;
}

__device__ int bin2dec(char *binary)
{
  int num,i,bin;
  num = 0;
  int j=0;
  for(i=RADIUS*2;i>=0;i--)
  {
    bin = (binary[i]=='0'?0:1);
    num += bin*powf(2,j++);
  }
  return num;
}

__device__ int executeCA(Individual* ind, Lattice lat)
{
  int pos,dif;
  int i,j,run;
  int finalDens;
  char bin[RADIUS*2+1];
  char next[LAT_SIZE];

  for(run=0;run<CA_RUNS;run++)
  {
    for(i=0;i<LAT_SIZE;i++)
    {
      dif = i-RADIUS;
      if(dif < 0) pos = LAT_SIZE+dif;
      else pos = dif;

      for(j=0;j<RADIUS*2+1;j++)
      {
        bin[j] = lat.cells[pos++];
        if(pos==LAT_SIZE)
          pos = 0;
      }
      next[i] = ind->rule[bin2dec(bin)];
    }
    
    for(j=0;j<LAT_SIZE;j++){ if(lat.cells[j]!=next[j]) break;}
    if(j==LAT_SIZE) break;

    for(i=0;i<LAT_SIZE;i++)
      lat.cells[i]=next[i];
  }

  finalDens=0;
  for(i=0;i<LAT_SIZE;i++)
  {
    if(lat.cells[i]=='1') 
      finalDens++;
  }

  if( (lat.density > LAT_SIZE/2 && finalDens == LAT_SIZE) ||
      (lat.density < LAT_SIZE/2 && finalDens == 0)          )
    ind->fitness++;
  return 0;
}

__global__ void cgaKernel(Individual* d_ind, Lattice* d_lat)
{
  int pInd = threadIdx.x;
  int lInd = 0;

  Lattice current;

  lInd = blockIdx.x;
  for(int i=0;i<LAT_SIZE;i++)
    current.cells[i] = d_lat[lInd].cells[i];
  current.density = d_lat[lInd].density;

  executeCA(&d_ind[pInd],current);
}

void crossOver(Individual *h_pop)
{
  Individual fat1,fat2,son1,son2;
  int f1Idx,f2Idx,s1Idx,s2Idx;
  int rest = POPULATION-CROSSAMOUT;
  int point = 0;
  int i,k;

  memset(son1.rule,'\0',RULE_SIZE);
  memset(son2.rule,'\0',RULE_SIZE);
  memset(fat1.rule,'\0',RULE_SIZE);
  memset(fat2.rule,'\0',RULE_SIZE);

  k=0;
  for(i=0;i<rest;i+=2)
  {
    f1Idx = rest + uniformDeviate(rand())* (POPULATION-rest-1);
    f2Idx = rest + uniformDeviate(rand())* (POPULATION-rest-1);

    point = 1+uniformDeviate(rand())*(RULE_SIZE-1); //we can't cross from point 0 (ie: crossover always happens)

    memcpy(&fat1,&h_pop[f1Idx],sizeof(Individual));
    memcpy(&fat2,&h_pop[f2Idx],sizeof(Individual));

    memcpy(&son1.rule,        &fat2.rule,        point);
    memcpy(&son1.rule[point], &fat1.rule[point], RULE_SIZE-point);
    memcpy(&son2.rule,        &fat1.rule,        point);
    memcpy(&son2.rule[point], &fat2.rule[point], RULE_SIZE-point);

    son1.rule[RULE_SIZE]='\0';
    son2.rule[RULE_SIZE]='\0';

    s1Idx = k++;
    s2Idx = k++;
    memcpy(&h_pop[s1Idx],&son1,sizeof(Individual));
    memcpy(&h_pop[s2Idx],&son2,sizeof(Individual));
  }
}

void mutate(Individual *h_pop)
{
  int i,j;
  for(i=0;i<(POPULATION-CROSSAMOUT);i++)
  {
    for(j=0;j<RULE_SIZE;j++)
    {
      if(uniformDeviate(rand()) <= MUTATE_PROB)
        h_pop[i].rule[j] = (h_pop[i].rule[j]=='0'?'1':'0');
    }
  }
}

