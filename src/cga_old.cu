#include "cga.h"
#include "consts.h"

#include <cuda.h>

__device__ int bin2dec(char *binary);
__device__ void executeCA(Individual* ind, Lattice *lat);
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
  fprintf(stderr,"Allocating...\n");
  d_error = cudaMalloc((void**)&d_pop,popSize);
  if(__printErr(stderr,d_error,"cudaMalloc()")!=0) return 1;
  d_error = cudaMalloc((void**)&d_lat,latSize);
  if(__printErr(stderr,d_error,"cudaMalloc()")!=0) return 1; 
  //Pass to device
  fprintf(stderr,"Passing to device...\n");
  d_error = cudaMemcpy(d_pop,h_pop,popSize,cudaMemcpyHostToDevice);
  if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;
  d_error = cudaMemcpy(d_lat,h_lat,latSize,cudaMemcpyHostToDevice);
  if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;

  //Call the Kernel
  fprintf(stderr,"Calling kernel...\n");
  dim3 grid(POPULATION);
  dim3 blocks(1);
  cudaThreadSynchronize();
  cgaKernel<<<grid,blocks>>>(d_pop,d_lat);
  cudaThreadSynchronize();

  d_error = cudaGetLastError();
  if(__printErr(stderr,d_error,"Kernel")!=0) return 1;
  //Copy back
  fprintf(stderr,"Passing to host...\n");
  d_error = cudaMemcpy(h_pop,d_pop,popSize,cudaMemcpyDeviceToHost);
  if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;
  d_error = cudaMemcpy(h_lat,d_lat,latSize,cudaMemcpyDeviceToHost);
  if(__printErr(stderr,d_error,"cudaMemcpy()")!=0) return 1;

/*
  for(int i=0;i<POPULATION;i++)
    printf("%d:%d\n",i,h_pop[i].fitness);
*/
  //Clean up
  fprintf(stderr,"Cleaning up...\n");
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

__device__ void executeCA(Individual* ind, Lattice *lat)
{
  int pos,dif;
  int i,j,run;
  int startDens,finalDens;
  char bin[RADIUS*2+1];
  char next[LAT_SIZE];
  char current[LAT_SIZE];

//  fprintf(stderr,".");
  startDens = finalDens = 0;
  for(i=0;i<LAT_SIZE;i++)
  {
    next[i] = '0';
    current[i] = lat->cells[i];
  }

  for(i=0;i<LAT_SIZE;i++)
  {
    if(current[i]=='1') startDens++;
  }

  for(run=0;run<CA_RUNS;run++)
  {
    for(i=0;i<LAT_SIZE;i++)
    {
      current[i] = next[i];
      next[i] = '0';
    }

    for(i=0;i<LAT_SIZE;i++)
    {
      dif = i-RADIUS;
      if(dif < 0) pos = LAT_SIZE+dif;
      else pos = dif;

      for(j=0;j<RADIUS*2+1;j++)
      {
        bin[j] = current[pos++];
        if(pos==LAT_SIZE)
          pos = 0;
      }
      next[i] = ind->rule[bin2dec(bin)];
    }
    
    for(j=0;j<LAT_SIZE;j++){ if(current[j]!=next[j]) break;}
    if(j==LAT_SIZE) break;

    for(i=0;i<LAT_SIZE;i++)
      current[i]=next[i];
  }

  for(i=0;i<LAT_SIZE;i++)
  {
    if(current[i]=='1') finalDens++;
  }

  if( (startDens > LAT_SIZE/2 && finalDens == LAT_SIZE) ||
      (startDens < LAT_SIZE/2 && finalDens == 0)          )
     ind->fitness++;
}

__global__ void cgaKernel(Individual* d_ind, Lattice* d_lat)
{
  int ind = blockIdx.x;
  if(ind < POPULATION)
  {    
    for(int i=0;i<MAX_LATS;i++)
      executeCA(&d_ind[ind],&d_lat[i]);
    //fprintf(stderr,"%d:%d\n",ind,d_ind[ind].fitness);
  }
}
