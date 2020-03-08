#ifndef FALOCAL_H
#define FALOCAL_H
#include "fxTypes.h"
#include "faSID.h"
#include "fxBase.h"
#include "faType.h"
#include "faSave.h"
#include "faLocal.h"


void faLocalTree(FaTreeNode** root, int depth, FaTrainData *traindata, int landmark_num, int node_position, int stage_num);
void faFreeTree(FaTreeNode** root);
#endif