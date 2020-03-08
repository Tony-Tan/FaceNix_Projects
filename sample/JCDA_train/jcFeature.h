#ifndef JCFEATURE_H
#define JCFEATURE_H
#include "fxTypes.h"
#include "fxBase.h"
#include "jcda.h"
#ifdef _OPENMP
#	include<omp.h>
#endif
typedef struct JcAlignmentFeature_ JcAlignmentFeature;
typedef struct JcDetectionFeature_ JcDetectionFeature;
typedef struct JcLeafFeature_  JcLeafFeature;

struct JcAlignmentFeature_{
	FxPoint off_set[2];
	int threshold;
	double min_variance;
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
	double min_entropy;
	/*
	less than threshold
	goto left child
	else goto right child
	*/
};

struct JcLeafFeature_{
	//JcLeafLinearParam off_set_linear_regression[27];
	//FxPoint64 off_set_array[27];
	double class_score;
};


JcAlignmentFeature* jcCreateAF_Array(uint32 size);
JcDetectionFeature* jcCreateDF_Array(uint32 size);
void jcFreeAF_Array(JcAlignmentFeature **af);
void jcFreeDF_Array(JcDetectionFeature **df);


uint32 jcA_DFeatureRadius(int cascade_stage);
void jcRandAlFeature(JcAlignmentFeature * a_f, uint32 size,uint32 radius);
void jcRandDeFeature(JcDetectionFeature * d_f, uint32 size,uint32 radius);
#endif