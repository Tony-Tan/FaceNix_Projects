#ifndef JCDA_H
#define JCDA_H

#include "fxTypes.h"
#include "fxBase.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#	include<omp.h>
#endif
#define TREE_DEPTH 4
#define OURPUTPRINT 1
#define LANDMARKTYPE 27
#define STAGES_OF_CASCADE 5
#define TOTAL_WEAK_C_R 135
#define WEAK_C_R_EACH_STAGE (TOTAL_WEAK_C_R/STAGES_OF_CASCADE)
#define OPENMP_THREAD_NUM 16

#define MIN_FEATURE_VALUE -256
#define MAX_FEATURE_VALUE 256
#define INTERAL_TEST 2000
//#define TOTAL_NEGATIVE_SAMPLE_BACK 20000

#define DEBUG 1
extern FILE * logfile;


typedef struct JcLandMark_{
	FxPoint64 LandMark[LANDMARKTYPE];
}JcLandMark;

typedef enum{
	NEGATIVE=-1,
	EMPTY=0,
	POSITIVE=1
}SampleLabel;

typedef struct JcTrainSample_{
	FxMat * mat;
	JcLandMark *GroundTruth;
	JcLandMark *Face_Shap;
	JcLandMark *Face_Shap_Delta;
	SampleLabel Label;
	double Class_Score;/*f_i*/
	double Sample_Weight;/*w_i*/
	int Position_in_Tree;/*0~15*/
	int isAvailable;
	int LBF[WEAK_C_R_EACH_STAGE];
}JcTrainSample;



typedef struct JcTrainData_{
	JcTrainSample * positive_sample;
	int positive_num;
	JcTrainSample * negative_sample;
	int negative_num;
}JcTrainData;










JcTrainData* jcCreateTrainData(int Positive_Num, int Negative_Num);
void jcFreeTrainData(JcTrainData** TrainData);
void jcReadTrainData(JcTrainData* TrainData, char * TrainDataPath);

#endif