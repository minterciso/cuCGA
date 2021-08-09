#include "kernel.h"

#include "consts.h"
__device__ void d_dec2bin(int decimal, char *bin, int size)
{
  int remain;
  do
  {
    remain = decimal%2;
    decimal = decimal/2;
    bin[size--] = (remain==0?'0':'1');
  }while(decimal>0);
}

__device__ int d_bin2dec(char *bin, int size)
{
  int i,n,sum=0;

  for(i=0;i<size;i++)
  {
    n = (bin[i]-'0') * powf(2,size-(i+1));
    sum+=n;
  }
  return sum;
}

__device__ void d_hex2bin(char *hex, char *bin, int h_size, int b_size)
{
  int i,k;
  k=0;
  for(i=0;i<h_size;i++)
  {
    if(hex[i]=='0') d_dec2bin(0,&bin[k],3);
    else if(hex[i]=='1') d_dec2bin(1,&bin[k],3);
    else if(hex[i]=='2') d_dec2bin(2,&bin[k],3);
    else if(hex[i]=='3') d_dec2bin(3,&bin[k],3);
    else if(hex[i]=='4') d_dec2bin(4,&bin[k],3);
    else if(hex[i]=='5') d_dec2bin(5,&bin[k],3);
    else if(hex[i]=='6') d_dec2bin(6,&bin[k],3);
    else if(hex[i]=='7') d_dec2bin(7,&bin[k],3);
    else if(hex[i]=='8') d_dec2bin(8,&bin[k],3);
    else if(hex[i]=='9') d_dec2bin(9,&bin[k],3);
    else if(hex[i]=='A' || hex[i]=='a') d_dec2bin(10,&bin[k],3);
    else if(hex[i]=='B' || hex[i]=='b') d_dec2bin(11,&bin[k],3);
    else if(hex[i]=='C' || hex[i]=='c') d_dec2bin(12,&bin[k],3);
    else if(hex[i]=='D' || hex[i]=='d') d_dec2bin(13,&bin[k],3);
    else if(hex[i]=='E' || hex[i]=='e') d_dec2bin(14,&bin[k],3);
    else if(hex[i]=='F' || hex[i]=='f') d_dec2bin(15,&bin[k],3);
    k+=4;
  }
}

__device__ void d_bin2hex(char *hex, char *bin, int h_size, int b_size)
{
  int i;
  int dec = 0;
  int pos = 0;

  for(i=0;i<h_size;i++)
  {
    dec = 0;
    dec=d_bin2dec(&bin[pos],4);
    switch(dec)
    {
      case 0:  hex[i]='0';break;
      case 1:  hex[i]='1';break;
      case 2:  hex[i]='2';break;
      case 3:  hex[i]='3';break;
      case 4:  hex[i]='4';break;
      case 5:  hex[i]='5';break;
      case 6:  hex[i]='6';break;
      case 7:  hex[i]='7';break;
      case 8:  hex[i]='8';break;
      case 9:  hex[i]='9';break;
      case 10: hex[i]='a';break;
      case 11: hex[i]='b';break;
      case 12: hex[i]='c';break;
      case 13: hex[i]='d';break;
      case 14: hex[i]='e';break;
      case 15: hex[i]='f';break;
    }
    pos+=4;
  }
}

__global__ void executeCA(Lattice *lat, char *rule)
{
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(t_idx < MAX_LATS)
  {
    int dif = 0;
    int pos = 0;
    char res[LAT_SIZE];
    char bin[RADIUS*2+1];
    int idx = 0;

    for(int i=0;i<CA_RUNS;i++)
    {
      memset(res,'0',LAT_SIZE);
      pos=0;
      dif=0;
      for(int j = 0; j < LAT_SIZE;j++)
      {
        dif = j - RADIUS;
        if(dif < 0) pos = LAT_SIZE+dif;
        else pos = dif;
        for(int k = 0; k < RADIUS*2+1;k++)
        {
          bin[k] = lat[t_idx].cells[pos];
          pos++;
          if(pos==LAT_SIZE) pos = 0;
        }
        idx = d_bin2dec(bin,RADIUS*2+1);
        if(idx >=0 && idx <RULE_SIZE)
          res[j] = rule[idx];
        memset(bin,'0',RADIUS*2+1);
      }
      memcpy(lat[t_idx].cells,res,LAT_SIZE);
    }
  }
}

