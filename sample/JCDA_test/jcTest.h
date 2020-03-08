#ifndef JCTEST_H
#define JCTEST_H

#include "fxTypes.h"
#include "fxBase.h"
#include "fxDIP.h"
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#	include<omp.h>
#endif
#define LANDMARKTYPE 27
#define ROI_WIDTH 40
#define ROI_HEIGHT 40
#define MAX_FEATURE_VALUE 256
#define MIN_FEATURE_VALUE -256
#include "jcRead.h"





typedef struct JcLandMark_{
	FxPoint64 LandMark[LANDMARKTYPE];
}JcLandMark;

typedef struct JcTestSample_{
	FxMat * mat;
	JcLandMark *Face_Shape;
	double Class_Score;
}JcTestSample;
typedef struct JcFace_ JcFace;
struct JcFace_{
	FxPoint face_center;
	FxSize face_size;
	JcLandMark Face_Shape;

	JcFace* next;
};



JcLandMark *jcReadMeanShape(char * filename);
JcTestSample * jcReadSample(FxMat* mat, JcLandMark * face_shape);

void jcFreeSample(JcTestSample ** sample);
void jcInitTest(char *cascade_file_path, char * mean_face_shape);
void jcReleaseTest();
JcFace* jcTest(FxMat *src);
#endif