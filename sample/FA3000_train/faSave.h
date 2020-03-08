#ifndef FASAVE_H
#define FASAVE_H
#include "faType.h"

void faSaveTree(FILE* file, FaTreeNode *root,int landmark);
void faSaveparam(char * path);
#endif