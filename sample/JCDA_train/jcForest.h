#ifndef JCFOREST_H
#define JCFOREST_H

#include "fxBase.h"
#include "jcda.h"
#include "jcFeature.h"
#include <float.h>



#define jcTreeLeftChild(position) ((position)*2+1)
#define jcTreeRightChild(position) ((position)*2+2)



typedef struct JcTreeNode_ JcTreeNode;
typedef enum JcTreeNodeType_ JcTreeNodeType;
typedef struct JcWeakCl_Re_ JcWeakCl_Re;
typedef struct JcSampleSet_ JcSampleSet;

enum JcTreeNodeType_{
	ALIGNMENT,
	DETECTION,
	LEAF
};


struct JcTreeNode_{
	JcTreeNodeType node_type;
	JcDetectionFeature * feature_dt;
	JcAlignmentFeature * feature_al;
	JcLeafFeature *feature_lf;
	JcTreeNode *left_child;
	JcTreeNode *right_child;
};

struct JcWeakCl_Re_{
	int landmark_num;/*which facial point to regress*/
	double class_threshold;
	JcTreeNode * root;
};

struct JcSampleSet_{
	int pos_size;
	int neg_size;
	int * pos_positions;
	int * neg_positions; 
};


void jcFreeTree(JcTreeNode * root);
void jcInitDataPose(JcTrainData *data);
//JcSampleSet* jcGetSample(JcTrainData * train_data, int position);
void jcLearnAlignmentNode(JcTrainData* train_data, int position, int cascade_stage, int facial_point, JcAlignmentFeature *feature);
void jcLearnDetectionNode(JcTrainData* train_data, int position, int cascade_stage, JcDetectionFeature *feature);
void jcLearnLeafNode(JcTrainData* train_data, int position, int face_point, int tree_num, JcLeafFeature *feature);
void jcUpdateTrainSample(JcTrainData* train_data, JcSampleSet *sampleset, double class_score);
void jcSplitAlignmentNode(JcTrainData*train_data, JcSampleSet *sample_set, int facial_point, int position, JcAlignmentFeature *al_feature);
JcSampleSet* jcSampleSet(JcTrainData * train_data, int position);
int jcDetectionFeatureValue(FxMat * mat, FxPoint * center, FxPoint * off_set);
JcTreeNode *jcCreateTreeNode(JcTreeNodeType node_type);
void jcFreeSampleSet(JcSampleSet** sampleset);
double jcClassificationScore(JcTrainData *train_data, JcSampleSet *sample_set);
void jcLeafOffset(JcTrainData *train_data, JcSampleSet * sample_set,FxPoint64 * offset);
int jcAlignmentFeatureValue(FxMat * mat, FxPoint64* center, FxPoint * off_set);
#endif