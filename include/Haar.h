#include "fxBase.h"
#include "fxError.h"
#include "AdaBoost.h"
typedef enum{
	FX_HAAR_TYPE1,
	FX_HAAR_TYPE2,
	FX_HAAR_TYPE3,
	FX_HAAR_TYPE4
}FxHaarType;
typedef struct FxHaarFeature_{
	FxPoint offset;
	FxSize	size;
	FxHaarType type;
}FxHaarFeature;
void fxIntegralImage(FxMat *src,FxMat *dst);
int fxHaar(FxMat *src, FxHaarFeature HaarFeature);




typedef struct FxHaarClassifier_{
	AdaBoostClassifier classifier;
	FxHaarFeature haar;
	double Haar_Threshold;
	double True_negitave_rate;
	struct FxHaarClassifier_ *next;
}FxHaarClassifier;


typedef struct FxCasCade_{
	int Classifier_size;
	FxHaarClassifier *Classifier_array;
	struct FxCasCade_* next;
}FxCasCade;
void fxHaarTrain(char *sample_path,char *cascadfilepath,int pos_num,int neg_num, FxSize size);