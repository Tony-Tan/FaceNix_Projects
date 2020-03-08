#ifndef FATRAIN_H
#define FATRAIN_H
#include "faType.h"
#include "faLocal.h"
#include "faLBF.h"
#include "faSave.h"
#include "faGlobal.h"
void faInitTarinData(FaTrainData *traindata);
void faTrain(FaTrainData *traindata,char * result_path);
#endif