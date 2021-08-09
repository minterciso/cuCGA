#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

void dec2bin(int decimal, char *bin, int size)
{
  int remain;
  do
  {
    remain = decimal%2;
    decimal = decimal/2;
    bin[size--] = (remain==0?'0':'1');
  }while(decimal>0);
}

int bin2dec(char *bin, int size)
{
  int i,n,sum=0;

  for(i=0;i<size;i++)
  {
    n = (bin[i]-'0') * pow(2,size-(i+1));
    sum+=n;
  }
  return sum;
}

void hex2bin(char *hex, char *bin, int h_size, int b_size)
{
  int i,k;
  k=0;
  for(i=0;i<h_size;i++)
  {
    if(hex[i]=='0') dec2bin(0,&bin[k],3);
    else if(hex[i]=='1') dec2bin(1,&bin[k],3);
    else if(hex[i]=='2') dec2bin(2,&bin[k],3);
    else if(hex[i]=='3') dec2bin(3,&bin[k],3);
    else if(hex[i]=='4') dec2bin(4,&bin[k],3);
    else if(hex[i]=='5') dec2bin(5,&bin[k],3);
    else if(hex[i]=='6') dec2bin(6,&bin[k],3);
    else if(hex[i]=='7') dec2bin(7,&bin[k],3);
    else if(hex[i]=='8') dec2bin(8,&bin[k],3);
    else if(hex[i]=='9') dec2bin(9,&bin[k],3);
    else if(hex[i]=='A' || hex[i]=='a') dec2bin(10,&bin[k],3);
    else if(hex[i]=='B' || hex[i]=='b') dec2bin(11,&bin[k],3);
    else if(hex[i]=='C' || hex[i]=='c') dec2bin(12,&bin[k],3);
    else if(hex[i]=='D' || hex[i]=='d') dec2bin(13,&bin[k],3);
    else if(hex[i]=='E' || hex[i]=='e') dec2bin(14,&bin[k],3);
    else if(hex[i]=='F' || hex[i]=='f') dec2bin(15,&bin[k],3);
    k+=4;
  }
}

void bin2hex(char *hex, char *bin, int h_size, int b_size)
{
  int i;
  int dec = 0;
  int pos = 0;

  for(i=0;i<h_size;i++)
  {
    dec = 0;
    dec=bin2dec(&bin[pos],4);
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

int timeSeed(void)
{
  time_t now = time (NULL);
  unsigned char *p = (unsigned char *)&now;
  int seed = 0;
  size_t i;
  for ( i = 0; i < sizeof(now); i++ )
    seed = seed * ( UCHAR_MAX + 2U ) + p[i];
  return seed;
}

double uniformDeviate ( int seed )
{
  return seed * ( 1.0 / ( RAND_MAX + 1.0 ) );
}

void bubbleSort(Individual *ind)
{
  int swapped = 0;
  int i = 0;
  Individual tmp;
  do
  {
    swapped=0;
    for(i=0;i<POPULATION-1;i++)
    {
      if(ind[i].fitness > ind[i+1].fitness)
      {
        memcpy(&tmp,     &ind[i],  sizeof(Individual));
        memcpy(&ind[i],  &ind[i+1],sizeof(Individual));
        memcpy(&ind[i+1],&tmp,     sizeof(Individual));
        swapped=1;
      }
    }
  }while(swapped==1);
}

