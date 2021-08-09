#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include "structs.h"

//converters
void dec2bin(int decimal, char *bin, int size);
int bin2dec(char *bin, int size);
void hex2bin(char *hex, char *bin, int h_size, int b_size);
void bin2hex(char *hex, char *bin, int h_size, int b_size);

//Random
int timeSeed(void);
double uniformDeviate ( int seed );

//Sorters
void bubbleSort(Individual *ind);

#endif //__UTILS_H
