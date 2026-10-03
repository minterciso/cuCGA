#include "params.h"

#include <stdlib.h>
#include <getopt.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include "consts.h"
#include "utils.h"

Params params = { DEFAULT_MUT_RATE, DEFAULT_CROSS_RATE, REP_BINARY, DEFAULT_T_MAX, DEFAULT_HASH_PROB, DEFAULT_GENERATIONS, 0, DEFAULT_N_ICS, NULL };

static const char *REP_NAMES[] = { "binary", "single", "double" };

static void usage(FILE *stream, const char *prog)
{
  fprintf(stream,
          "Usage: %s [options]\n"
          "  -m, --mutation-rate R   per-symbol mutation probability in [0,1]\n"
          "                          (default %g for binary, %g for single/double)\n"
          "  -c, --crossover-rate R  crossover probability p_c in [0,1]\n"
          "                          (default %g; MCH uses 0.8, CMD 1.0)\n"
          "  -r, --representation S  binary, single or double (default binary).\n"
          "                          single/double use ternary templates with single or\n"
          "                          double orientation\n"
          "  -t, --t-max N           maximum initial templates per individual, 0..%d\n"
          "                          (default %d)\n"
          "  -p, --hash-prob P       probability of '#' in each template cell, in [0,1]\n"
          "                          (default %g)\n"
          "  -g, --generations N     generations of the GA (default %d)\n"
          "  -s, --seed N            random seed, 0..%u (default: derived from the clock)\n"
          "  -n, --ics N             binomial ICs for the final evaluation (default %d)\n"
          "  -v, --validate HEX      only evaluate the given %d-digit hex rule (neighbourhood\n"
          "                          0000000 first, as in MCH/CMD) on N binomial ICs, no GA\n"
          "  -h, --help              show this help\n",
          prog, DEFAULT_MUT_RATE, DEFAULT_TPL_MUT_RATE, DEFAULT_CROSS_RATE, MAX_TEMPLATES, DEFAULT_T_MAX, DEFAULT_HASH_PROB, DEFAULT_GENERATIONS, UINT_MAX, DEFAULT_N_ICS, RULE_SIZE/4);
}

static int parseDouble(const char *s, double min, double max, double *out)
{
  char *end = NULL;
  double v = strtod(s, &end);
  if(end == s || *end != '\0' || v < min || v > max)
    return -1;
  *out = v;
  return 0;
}

static int parseUInt(const char *s, unsigned int *out)
{
  char *end = NULL;
  unsigned long long v;
  if(*s < '0' || *s > '9') //Rejects signs and leading whitespace, which strtoull accepts
    return -1;
  errno = 0;
  v = strtoull(s, &end, 10);
  if(end == s || *end != '\0' || errno == ERANGE || v > UINT_MAX)
    return -1;
  *out = (unsigned int)v;
  return 0;
}

int parseParams(int argc, char *argv[])
{
  static const struct option opts[] =
  {
    {"mutation-rate",  required_argument, NULL, 'm'},
    {"crossover-rate", required_argument, NULL, 'c'},
    {"representation", required_argument, NULL, 'r'},
    {"t-max",          required_argument, NULL, 't'},
    {"hash-prob",      required_argument, NULL, 'p'},
    {"generations",    required_argument, NULL, 'g'},
    {"seed",           required_argument, NULL, 's'},
    {"ics",            required_argument, NULL, 'n'},
    {"validate",       required_argument, NULL, 'v'},
    {"help",           no_argument,       NULL, 'h'},
    {NULL, 0, NULL, 0}
  };
  int opt,i;
  int seed_set = 0;
  int mut_set = 0;
  unsigned int u;

  while((opt = getopt_long(argc, argv, "m:c:r:t:p:g:s:n:v:h", opts, NULL)) != -1)
  {
    switch(opt)
    {
      case 'm':
        if(parseDouble(optarg, 0.0, 1.0, &params.mut_rate) != 0)
        {
          fprintf(stderr, "Invalid mutation rate '%s': expected a number in [0,1]\n", optarg);
          return -1;
        }
        mut_set = 1;
        break;
      case 'c':
        if(parseDouble(optarg, 0.0, 1.0, &params.cross_rate) != 0)
        {
          fprintf(stderr, "Invalid crossover rate '%s': expected a number in [0,1]\n", optarg);
          return -1;
        }
        break;
      case 'r':
        for(i=0;i<3 && strcmp(optarg, REP_NAMES[i])!=0;i++);
        if(i == 3)
        {
          fprintf(stderr, "Invalid representation '%s': expected binary, single or double\n", optarg);
          return -1;
        }
        params.representation = (Representation)i;
        break;
      case 't':
        if(parseUInt(optarg, &u) != 0 || u > MAX_TEMPLATES)
        {
          fprintf(stderr, "Invalid t-max '%s': expected an integer in [0,%d]\n", optarg, MAX_TEMPLATES);
          return -1;
        }
        params.t_max = (int)u;
        break;
      case 'p':
        if(parseDouble(optarg, 0.0, 1.0, &params.hash_prob) != 0)
        {
          fprintf(stderr, "Invalid hash probability '%s': expected a number in [0,1]\n", optarg);
          return -1;
        }
        break;
      case 's':
        if(parseUInt(optarg, &params.seed) != 0)
        {
          fprintf(stderr, "Invalid seed '%s': expected an integer in [0,%u]\n", optarg, UINT_MAX);
          return -1;
        }
        seed_set = 1;
        break;
      case 'g':
        if(parseUInt(optarg, &u) != 0 || u == 0 || u > INT_MAX)
        {
          fprintf(stderr, "Invalid number of generations '%s': expected a positive integer\n", optarg);
          return -1;
        }
        params.generations = (int)u;
        break;
      case 'n':
        if(parseUInt(optarg, &u) != 0 || u == 0 || u > INT_MAX)
        {
          fprintf(stderr, "Invalid number of ICs '%s': expected a positive integer\n", optarg);
          return -1;
        }
        params.n_ics = (int)u;
        break;
      case 'v':
        params.validate_hex = optarg;
        break;
      case 'h':
        usage(stdout, argv[0]);
        return 1;
      default:
        usage(stderr, argv[0]);
        return -1;
    }
  }
  if(optind < argc)
  {
    fprintf(stderr, "Unexpected argument '%s'\n", argv[optind]);
    usage(stderr, argv[0]);
    return -1;
  }
  if(!mut_set && params.representation != REP_BINARY)
    params.mut_rate = DEFAULT_TPL_MUT_RATE;
  if(!seed_set)
    params.seed = (unsigned int)timeSeed();
  return 0;
}

void printParams(FILE *stream)
{
  fprintf(stream, "mutation-rate=%g crossover-rate=%g representation=%s",
          params.mut_rate, params.cross_rate, REP_NAMES[params.representation]);
  if(params.representation != REP_BINARY)
    fprintf(stream, " t-max=%d hash-prob=%g", params.t_max, params.hash_prob);
  fprintf(stream, " generations=%d seed=%u ics=%d\n", params.generations, params.seed, params.n_ics);
}
