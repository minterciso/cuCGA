#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include "structs.h"

//converters
void dec2bin(int decimal, char *bin, int size);
int bin2dec(char *bin, int size);
void hex2bin(char *hex, char *bin, int h_size, int b_size);
void bin2hex(char *hex, char *bin, int h_size, int b_size);
//Decimal rule number as in the CEC 2011 paper: sum of rule[k]*2^k
#define RULE_DEC_SIZE 40 //2^128-1 has 39 digits
void ruleToDecimal(const char *rule, char *out);
//Parses a RULE_SIZE/4-digit hex rule (neighbourhood 0000000 first) into rule; 0 if invalid
int parseRule(const char *hex, char *rule);

//Random
int timeSeed(void);
double uniformDeviate ( int seed );

//Sorters
void bubbleSort(Individual *ind, int n);

#endif //__UTILS_H
