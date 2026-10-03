#include "ca.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void createRandomLattices(Individual *ind)
{
  assert(ind!=NULL);
  int j;
  int count=0;
  int rnd=0;

  for(j=0;j<MAX_LATS;j++)
  {
    memset(ind->lat[j].cells,'0',LAT_SIZE); //We allways start with an empty Lattice
#ifndef VALIDATE
    ind->lat[j].density = uniformDeviate(rand())*(LAT_SIZE+1); //Uniform distribution over [0,LAT_SIZE]
    count=0;
    while(count < ind->lat[j].density)
    {
      rnd = uniformDeviate(rand())*LAT_SIZE; //All cells have equal probability to be choosen
      if(ind->lat[j].cells[rnd]=='0')
      {
        ind->lat[j].cells[rnd]='1';
        count++;
      }
    }
#endif
#ifdef VALIDATE
    //Unbiased distribution: each cell is 1 with probability 0.5, so the density is Binomial(LAT_SIZE,0.5)
    count=0;
    for(rnd=0;rnd<LAT_SIZE;rnd++)
    {
      if(uniformDeviate(rand()) < 0.5)
      {
        ind->lat[j].cells[rnd]='1';
        count++;
      }
    }
    ind->lat[j].density = count;
#endif
  }
}

void createRandomRules(Individual *ind)
{
  int i;
  int rnd = 0;

  memset(ind->rule,'0',RULE_SIZE);
  for(i=0;i<RULE_SIZE;i++)
  {
    rnd = uniformDeviate(rand())*2;
    switch(rnd)
    {
      case 0:ind->rule[i]='0';break;
      case 1:ind->rule[i]='1';break;
    }
  }
}

void createUnbiasedLattices(Lattice *lat, int n)
{
  int i,k;
  for(i=0;i<n;i++)
  {
    lat[i].density = 0;
    for(k=0;k<LAT_SIZE;k++)
    {
      lat[i].cells[k] = (uniformDeviate(rand()) < 0.5 ? '1' : '0');
      if(lat[i].cells[k]=='1') lat[i].density++;
    }
  }
}

int classify(const Lattice *lat, const char *rule, int *fixed)
{
  int k,count = 0;
  for(k=0;k<LAT_SIZE;k++)
    if(lat->cells[k]=='1') count++;
  //LAT_SIZE is odd, so density <= LAT_SIZE/2 means a majority of 0s
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

//Lattice t uses rule t/latsPerRule (RULE_SIZE chars each)
void cpuRunCA(Lattice *lat, const char *rules, int nLats, int latsPerRule)
{
  int t;
  for(t=0;t<nLats;t++)
    executeCA(&lat[t],&rules[(t/latsPerRule)*RULE_SIZE],t%latsPerRule,t/latsPerRule);
}

void executeCA(Lattice *lat, const char *rule, int ind_idx, int th_idx)
{
  int dif = 0;
  int pos = 0;
  int i,j,k;
  char res[LAT_SIZE];
  char bin[RADIUS*2+1+1];
  int idx=0;

  memset(bin,'0',RADIUS*2+1);

#ifdef DEBUG
  char fname[FNAME_SIZE];
  memset(fname,'\0',sizeof(FNAME));
  snprintf(fname,FNAME_SIZE-1,"logs/individual%03d-%03d.log",th_idx,ind_idx);
  FILE *fp = fopen(fname,"a+");
#endif
  for(i=0;i<CA_RUNS;i++)
  {
    memset(res,'0',LAT_SIZE);
#ifdef DEBUG
    fprintf(fp,"%3d:",i);
    printCA(fp,lat,1);
    fprintf(fp,"\n");
#endif
    for(j=0;j<LAT_SIZE;j++)
    {
      dif = j-RADIUS;
      if(dif < 0)
        pos = LAT_SIZE+dif;
      else
        pos = dif;
 
      for(k=0;k<RADIUS*2+1;k++)
      {
        //bin[k]= (lat->cells[pos]=='0'?'1':'0');
        bin[k]=lat->cells[pos];
        pos++;
        if(pos==LAT_SIZE)
          pos = 0;
      }
      bin[RADIUS*2+1]='\0';
      idx = bin2dec(bin,RADIUS*2+1);
      if(idx>=0 && idx<RULE_SIZE)
        res[j] = rule[idx];
      else
      {
        fprintf(stderr,"\nSOMETHING IS VERY WRONG!(%s)idx=%d th_idx=%d id_idx=%d\n",bin,idx,th_idx,ind_idx);
        abort();
      }
      memset(bin,'0',RADIUS*2+1);
    }
    if(memcmp(lat->cells,res,LAT_SIZE)==0) break; //If we hang...stop
    memcpy(lat->cells,res,LAT_SIZE);
  }
#ifdef DEBUG
  fclose(fp);
#endif
}

void printCA(FILE *stream, Lattice *lat, int mode)
{
  int i;
  for(i=0;i<LAT_SIZE;i++)
  {
    if(mode==0)
      fprintf(stream,"%c", (lat->cells[i]=='0'?'#':' ') );
    else if(mode==1)
      fprintf(stream,"%c", (lat->cells[i]=='0'?' ':'#') );
    else
      fprintf(stream,"%c",lat->cells[i]);
  }
}

