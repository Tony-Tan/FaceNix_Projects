#ifndef JCTRAIN_H
#define JCTRAIN_H

#include "jcda.h"
#include "jcForest.h"
#include "jcRealBoost.h"
#include "jcNegSampleMining.h"
#include "jcGlobal.h"
void * jcInitFaceShape(char *file_path,JcTrainData *train_data);

void jcTrainJCDA(JcTrainData *train_data,char *forest_file_path);
int jcLearnATree(JcTreeNode **root, JcTrainData *train_data, int cascade_stage, int tree_num, int tree_depth, int node_position);
JcTreeNodeType jcNodeType(int cascade_stage);
void jcLearnWeak_CL_RE(JcTrainData *train_data, int face_point, int cascade_stage, JcWeakCl_Re * weak_cl_re);
void jcSaveCl_ReFile(FILE * file, JcWeakCl_Re * weak);
#endif