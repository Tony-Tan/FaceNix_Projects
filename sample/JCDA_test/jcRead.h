#ifndef JCREAD_H
#define JCREAD_H

#include "fxTypes.h"


#define LANDMARKTYPE 27
#define STAGES_OF_CASCADE 5
#define TOTAL_WEAK_C_R 5400
#define WEAK_C_R_EACH_STAGE (TOTAL_WEAK_C_R / STAGES_OF_CASCADE)




typedef struct JcAlignmentFeature_ JcAlignmentFeature;
typedef struct JcDetectionFeature_ JcDetectionFeature;
typedef struct JcLeafFeature_  JcLeafFeature;
typedef struct JcTreeNode_ JcTreeNode;
typedef enum JcTreeNodeType_ JcTreeNodeType;
typedef struct JcCascadeData_ JcCascadeData;
typedef struct JcWeakC_R_ JcWeakC_R;
struct JcAlignmentFeature_{
	FxPoint off_set[2];
	int threshold;
	/*
	less than threshold
	goto left child
	else goto right child
	*/
};
struct JcDetectionFeature_{
	int FacialPoint[2];/*from 0 to 28*/
	FxPoint off_set[2];
	int threshold;
	/*
	less than threshold
	goto left child
	else goto right child
	*/
};
struct JcLeafFeature_{
	FxPoint64 off_set_array[27];
	double class_score;
};
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
struct JcWeakC_R_{
	JcTreeNode * root;
	int landmark_num;
	double threshold;
};
struct JcCascadeData_{
	JcWeakC_R CascadeData[STAGES_OF_CASCADE][WEAK_C_R_EACH_STAGE];
};


JcCascadeData * jcReadCascadeData(char * path);
void jcFreeCascadeData(JcCascadeData **casdata);
#endif