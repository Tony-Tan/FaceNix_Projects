#ifndef FAREAD_H
#define FAREAD_H
#include "faType.h"
#ifdef _OPENMP
#include <omp.h>
#endif


FaTrainData * faReadTrainData(char * traindata_path, int traindata_size, char *mean_shape_path);
FaTrainData * faCreateTrainData(int data_size);
void faFreeTrainData(FaTrainData **traindata);
void faDeltaLandmark(FaTrainData* traindata, char * mean_shape_path);
#endif