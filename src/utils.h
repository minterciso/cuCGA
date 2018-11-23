#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "structs.h"

int bin2dec(char *binary);
void bin2hex(char *hex, char *bin, int h_size, int b_size);
void hex2bin(char *hex, char *bin, int h_size, int b_size);

int timeSeed(void);
double uniformDeviate(int seed);

void bubbleSort(Individual *h_pop);

void debug(const char *format, ...);

#endif //__UTILS_H

