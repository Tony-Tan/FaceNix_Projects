#ifndef FAWRITE_H
#define FAWRITE_H
#include "faType.h"
#ifdef _OPENMP
#include <omp.h>
#endif


void faWriteData(FaForest * forest);
#endif