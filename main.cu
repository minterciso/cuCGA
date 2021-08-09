#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

#include "structs.h"
#include "consts.h"
#include "ca.h"
#include "cga.h"
#include "utils.h"
#include "kernel.h"

int main(int argc, char *argv[])
{
  Individual *h_pop;
  Lattice *d_lat;
  char *d_rule;
  size_t h_memSize = sizeof(Individual)*POPULATION;
  size_t d_memSize = sizeof(Lattice)*MAX_LATS;
  cudaError_t d_error = cudaSuccess;

  srand(timeSeed());
  //Allocate host memory
  if((h_pop = (Individual*)malloc(h_memSize))==NULL)
  {
    perror("malloc");
    return EXIT_FAILURE;
  }
  memset(h_pop,'\0',h_memSize);

  //Allocating device memory
  d_error = cudaMalloc((void**)&d_lat,d_memSize);
  if(d_error!=cudaSuccess)
  {
    fprintf(stderr,"cudaMalloc:%s\n",cudaGetErrorString(d_error));
    free(h_pop);
    return EXIT_FAILURE;
  }
  d_error = cudaMalloc((void**)&d_rule,RULE_SIZE);
  if(d_error!=cudaSuccess)
  {
    fprintf(stderr,"cudaMalloc:%s\n",cudaGetErrorString(d_error));
    cudaFree(d_lat);
    free(h_pop);
    return EXIT_FAILURE;
  }

  for(int i = 0; i < POPULATION; i++)
  {
    createRandomLattice(&h_pop[i]);
#ifndef USE_BEST
    createRandomRules(&h_pop[i]);
#endif
#ifdef USE_BEST
    memset(h_pop[i].rule,'0',RULE_SIZE);
    hex2bin(BEST_CGA,h_pop[i].rule,32,RULE_SIZE);
#endif
  }

  FILE *fp = fopen("logs/output.log","w+");

  for(int g=0;g<GA_RUNS;g++)
  {
    fprintf(stderr,".");
    fprintf(fp,"Run %3d:",g);
    //Calling the Kernel (for all individuals, this is LAME)
    for(int i = 0; i < POPULATION;i++)
    {
      //Passing rule to device
      d_error = cudaMemcpy(d_rule,h_pop[0].rule,RULE_SIZE,cudaMemcpyHostToDevice);
      if(d_error!=cudaSuccess)
      {
        fprintf(stderr,"cudaMemcpy (h2d):%s\n",cudaGetErrorString(d_error));
        cudaFree(d_lat);
        cudaFree(d_rule);
        free(h_pop);
        return EXIT_FAILURE;
      }
      //Passing Lattice to device
      d_error = cudaMemcpy(d_lat,h_pop[i].lat,d_memSize,cudaMemcpyHostToDevice);
      if(d_error!=cudaSuccess)
      {
        fprintf(stderr,"cudaMemcpy (h2d):%s\n",cudaGetErrorString(d_error));
        cudaFree(d_lat);
        cudaFree(d_rule);
        free(h_pop);
        return EXIT_FAILURE;
      }

      int count = 0;
      dim3 gridQtd(512);
      dim3 gridSize(128);
      executeCA<<<gridQtd,gridSize>>>(d_lat,d_rule);
      cudaDeviceSynchronize();

      d_error = cudaMemcpy(h_pop[i].lat,d_lat,d_memSize,cudaMemcpyDeviceToHost);
      if(d_error!=cudaSuccess)
      {
        fprintf(stderr,"cudaMemcpy (d2h):%s\n",cudaGetErrorString(d_error));
        cudaFree(d_lat);
        cudaFree(d_rule);
        free(h_pop);
        return EXIT_FAILURE;
      }

      for(int j = 0;j <MAX_LATS;j++)
      {
        count = 0;
        for(int k=0;k<LAT_SIZE;k++)
          if(h_pop[i].lat[j].cells[k]=='1') count++;
        if( (h_pop[i].lat[j].density > LAT_SIZE/2 && count==LAT_SIZE) ||
            (h_pop[i].lat[j].density < LAT_SIZE/2 && count==0))
          h_pop[i].fitness++;
      }
      fprintf(fp,"%3d ",h_pop[i].fitness);
    }
    bubbleSort(h_pop);
    fprintf(fp,"(%3d)\n",h_pop[POPULATION-1].fitness);
    char hex[32];
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
  fclose(fp);
  //Clear memory
  free(h_pop);
  cudaFree(d_lat);
    cudaFree(d_rule);

  return EXIT_SUCCESS;
}
