#ifndef FAGLOBAL_H
#define FAGLOBAL_H
#include "jcda.h"
#include "linear.h"
void jcGlobal(JcTrainData *train_data, int stage_num, char* global_file_nam);
void jcUpdateShape(JcTrainData *train_data,int stage_num);
void jcUpdateSampleShape(JcTrainSample *data_sample, int stage_num, int isNeg);
#endif