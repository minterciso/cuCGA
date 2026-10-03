#include <stdio.h>
#include <string.h>

#include "params.h"
#include "templates.h"
#include "utils.h"

static int failures = 0;

#define CHECK(cond, ...) do { if(!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); failures++; } } while(0)

static Individual ind; //Too large for comfort on the stack

static void setTemplates(const char *tpls[], int n, char orientation)
{
  int t;
  memset(&ind, 0, sizeof(ind));
  ind.n_tpl = n;
  ind.orientation = orientation;
  for(t=0;t<n;t++)
    memcpy(ind.tpl[t].cells, tpls[t], NEIGH_SIZE);
}

static void checkRule(const char *expected, const char *what)
{
  char dec[RULE_DEC_SIZE];
  decodeTemplates(&ind);
  ruleToDecimal(ind.rule, dec);
  CHECK(strcmp(dec, expected) == 0, "%s: got %s, expected %s", what, dec, expected);
}

int main(void)
{
  int t,c,changed;
  Individual a,b;

  //GKL as 3 single-orientation templates must be rule MCHs of Table III (CEC 2011)
  const char *gkl[] = { "###11##", "###1##1", "1#10###" };
  setTemplates(gkl, 3, '1');
  checkRule("333636105325236971337806416870490831360", "GKL");

  //One all-'#' template matches everything
  const char *all[] = { "#######" };
  setTemplates(all, 1, '1');
  checkRule("340282366920938463463374607431768211455", "all-# template");

  //No templates: everything maps to the complement of the orientation
  setTemplates(NULL, 0, '1');
  checkRule("0", "empty, orientation 1");
  setTemplates(NULL, 0, '0');
  checkRule("340282366920938463463374607431768211455", "empty, orientation 0");

  //Single fully specified neighbourhood 0000001 -> only rule[1] set
  const char *one[] = { "0000001" };
  setTemplates(one, 1, '1');
  checkRule("2", "neighbourhood 0000001");

  //Swapping keeps template counts and moves one template each way
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  a.n_tpl = 1; memcpy(a.tpl[0].cells, "0000000", NEIGH_SIZE);
  b.n_tpl = 1; memcpy(b.tpl[0].cells, "1111111", NEIGH_SIZE);
  swapTemplates(&a, &b);
  CHECK(a.n_tpl == 1 && b.n_tpl == 1, "swap changed template counts");
  CHECK(memcmp(a.tpl[0].cells, "1111111", NEIGH_SIZE) == 0 && memcmp(b.tpl[0].cells, "0000000", NEIGH_SIZE) == 0, "swap did not exchange templates");

  //Mutation with rate 1 changes every cell to a different symbol of {0,1,#}
  params.mut_rate = 1.0;
  const char *mix[] = { "01#01#0", "#######" };
  setTemplates(mix, 2, '1');
  mutateTemplates(&ind);
  changed = 1;
  for(t=0;t<2;t++)
    for(c=0;c<NEIGH_SIZE;c++)
      if(ind.tpl[t].cells[c] == mix[t][c] || strchr("01#", ind.tpl[t].cells[c]) == NULL)
        changed = 0;
  CHECK(changed, "mutation with rate 1 left a cell unchanged or invalid");

  //Mutation with rate 0 changes nothing
  params.mut_rate = 0.0;
  setTemplates(mix, 2, '1');
  mutateTemplates(&ind);
  CHECK(memcmp(ind.tpl[0].cells, mix[0], NEIGH_SIZE) == 0 && memcmp(ind.tpl[1].cells, mix[1], NEIGH_SIZE) == 0, "mutation with rate 0 changed a cell");

  //Random templates respect t_max, orientation and the symbol set
  params.t_max = 9;
  params.hash_prob = 2.0/7.0;
  for(t=0;t<1000;t++)
  {
    params.representation = (t%2 ? REP_DOUBLE : REP_SINGLE);
    createRandomTemplates(&ind);
    CHECK(ind.n_tpl >= 0 && ind.n_tpl <= params.t_max, "n_tpl %d out of [0,%d]", ind.n_tpl, params.t_max);
    CHECK(params.representation == REP_DOUBLE || ind.orientation == '1', "single orientation must be '1'");
    CHECK(ind.orientation == '0' || ind.orientation == '1', "invalid orientation %c", ind.orientation);
    for(c=0;c<ind.n_tpl*NEIGH_SIZE;c++)
      CHECK(strchr("01#", ind.tpl[c/NEIGH_SIZE].cells[c%NEIGH_SIZE]) != NULL, "invalid template symbol");
  }

  if(failures == 0)
    printf("All template tests passed\n");
  return failures == 0 ? 0 : 1;
}
