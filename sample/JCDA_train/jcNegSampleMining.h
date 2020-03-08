#ifndef JCNEGSAMPLEMINING_H
#define JCNEGSAMPLEMINING_H
#include "jcda.h"
#include "jcForest.h"
#include "fxDIP.h"
#include "jcGlobal.h"
#define SAMPLE_WIDTH 40
#define SAMPLE_HEIGHT 40
#define TOTALNEGSIZE 20000
extern char * TrainData_Path;
extern JcLandMark MEANSHAPE;
typedef struct JcRemovedSampleSet_ JcRemovedSampleSet;

struct JcRemovedSampleSet_{
	int removed_size;
	int *removed_position;
};
JcRemovedSampleSet *jcGetRemovedNegSample(JcTrainData *train_data);
void jcFreeRemovedNeg_Sample(JcRemovedSampleSet ** sampleset);
void jcMiningNegSample(JcTrainData *train_data, JcWeakCl_Re * cl_re, int size);

#endif