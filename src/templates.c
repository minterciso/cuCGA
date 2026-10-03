#include "templates.h"

#include <stdlib.h>
#include "params.h"
#include "utils.h"

static const char SYMBOLS[] = "01#";

void createRandomTemplates(Individual *ind)
{
  int t,c;

  ind->n_tpl = uniformDeviate(rand()) * (params.t_max+1);
  for(t=0;t<ind->n_tpl;t++)
  {
    for(c=0;c<NEIGH_SIZE;c++)
    {
      if(uniformDeviate(rand()) < params.hash_prob)
        ind->tpl[t].cells[c] = '#';
      else
        ind->tpl[t].cells[c] = (uniformDeviate(rand()) < 0.5 ? '0' : '1');
    }
  }
  if(params.representation == REP_DOUBLE)
    ind->orientation = (uniformDeviate(rand()) < 0.5 ? '0' : '1');
  else
    ind->orientation = '1';
}

void decodeTemplates(Individual *ind)
{
  int k,t,c,hit;
  char bit;
  char other = (ind->orientation == '1' ? '0' : '1');

  //k is the neighbourhood value with cells[0] as the most significant bit, the
  //same order bin2dec() uses in executeCA()
  for(k=0;k<RULE_SIZE;k++)
  {
    hit = 0;
    for(t=0;t<ind->n_tpl && !hit;t++)
    {
      for(c=0;c<NEIGH_SIZE;c++)
      {
        bit = ((k >> (NEIGH_SIZE-1-c)) & 1) ? '1' : '0';
        if(ind->tpl[t].cells[c] != '#' && ind->tpl[t].cells[c] != bit)
          break;
      }
      hit = (c == NEIGH_SIZE);
    }
    ind->rule[k] = (hit ? ind->orientation : other);
  }
}

void swapTemplates(Individual *a, Individual *b)
{
  int ta = uniformDeviate(rand()) * a->n_tpl;
  int tb = uniformDeviate(rand()) * b->n_tpl;
  Template tmp = a->tpl[ta];

  a->tpl[ta] = b->tpl[tb];
  b->tpl[tb] = tmp;
}

void mutateTemplates(Individual *ind)
{
  int t,c,cur,shift;

  for(t=0;t<ind->n_tpl;t++)
  {
    for(c=0;c<NEIGH_SIZE;c++)
    {
      if(uniformDeviate(rand()) < params.mut_rate)
      {
        cur = (ind->tpl[t].cells[c] == '#' ? 2 : ind->tpl[t].cells[c]-'0');
        shift = 1 + (int)(uniformDeviate(rand()) * 2);
        ind->tpl[t].cells[c] = SYMBOLS[(cur+shift) % 3];
      }
    }
  }
}
