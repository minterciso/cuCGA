#include "kernel.h"

#include "consts.h"
#include "backend.h"

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t _err = (call);                                                     \
    if(_err != cudaSuccess)                                                        \
    {                                                                              \
      fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_err)); \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  } while(0)

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define CELLS_PER_LANE ((LAT_SIZE+WARP_SIZE-1)/WARP_SIZE)
#define NEIGH (2*RADIUS+1)

//Per-warp shared state: the lattice, double buffered, as 0/1 bytes, and the warp's rule
typedef struct WarpState
{
  unsigned char cells[2][LAT_SIZE];
  unsigned char rule[RULE_SIZE];
}WarpState;

//One warp per lattice; lattice w uses rule w/latsPerRule (RULE_SIZE chars each).
//Lane l owns the contiguous cells [l*CELLS_PER_LANE, (l+1)*CELLS_PER_LANE) and computes
//them with a sliding window over the neighbourhood: the index of cell j is the
//NEIGH-bit number formed by cells j-RADIUS..j+RADIUS, j-RADIUS being the most significant
//bit (as bin2dec() in the CPU version). The warp runs lat[w].steps steps and stops early at
//a fixed point, like the CPU version, which gives the same final lattice as running them all.
__global__ void executeCAKernel(Lattice *lat, const char *rules, int nLats, int latsPerRule)
{
  __shared__ WarpState state[WARPS_PER_BLOCK];
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int w = blockIdx.x*WARPS_PER_BLOCK + warp;
  if(w >= nLats)
    return; //The whole warp leaves together

  WarpState *s = &state[warp];
  char *cells = lat[w].cells;
  const char *rule = &rules[(size_t)(w/latsPerRule)*RULE_SIZE];
  int first = lane*CELLS_PER_LANE;
  int last  = min(first+CELLS_PER_LANE, LAT_SIZE); //first >= last for idle lanes
  int cur = 0;
  unsigned int steps = lat[w].steps;

  for(int j=lane;j<LAT_SIZE;j+=WARP_SIZE)
    s->cells[0][j] = (cells[j]=='1');
  for(int k=lane;k<RULE_SIZE;k+=WARP_SIZE)
    s->rule[k] = (rule[k]=='1');
  __syncwarp();

  for(unsigned int step=0;step<steps;step++)
  {
    const unsigned char *in = s->cells[cur];
    unsigned char *out = s->cells[cur^1];
    int changed = 0;
    if(first < last)
    {
      //Prime the window with cells first-RADIUS-1 .. first+RADIUS-1
      unsigned int idx = 0;
      int pos = first-RADIUS-1+LAT_SIZE;
      for(int k=0;k<NEIGH-1;k++)
      {
        pos++;
        if(pos>=LAT_SIZE) pos-=LAT_SIZE;
        idx = (idx<<1) | in[pos];
      }
      for(int j=first;j<last;j++)
      {
        pos++;
        if(pos>=LAT_SIZE) pos-=LAT_SIZE;
        idx = ((idx<<1) | in[pos]) & (RULE_SIZE-1);
        unsigned char v = s->rule[idx];
        changed |= (v != in[j]);
        out[j] = v;
      }
    }
    __syncwarp();
    if(!__any_sync(0xffffffffu, changed))
      break; //Fixed point: out equals in, keep in as the final state
    cur ^= 1;
  }

  for(int j=lane;j<LAT_SIZE;j+=WARP_SIZE)
    cells[j] = (s->cells[cur][j] ? '1' : '0');
}

//Backend entry point (backend.h)
extern "C" void runCA(Lattice *h_lat, const char *h_rules, int nLats, int latsPerRule)
{
  Lattice *d_lat;
  char *d_rules;
  size_t latSize  = sizeof(Lattice)*nLats;
  size_t ruleSize = (size_t)RULE_SIZE*((nLats+latsPerRule-1)/latsPerRule);

  CUDA_CHECK(cudaMalloc((void**)&d_lat,latSize));
  CUDA_CHECK(cudaMalloc((void**)&d_rules,ruleSize));
  CUDA_CHECK(cudaMemcpy(d_lat,h_lat,latSize,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rules,h_rules,ruleSize,cudaMemcpyHostToDevice));

  dim3 blockSize(WARPS_PER_BLOCK*WARP_SIZE);
  dim3 gridSize((nLats+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK);
  executeCAKernel<<<gridSize,blockSize>>>(d_lat,d_rules,nLats,latsPerRule);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_lat,d_lat,latSize,cudaMemcpyDeviceToHost));
  cudaFree(d_lat);
  cudaFree(d_rules);
}
