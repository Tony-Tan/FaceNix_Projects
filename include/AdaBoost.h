#ifndef ADABOOST_H
#define ADABOOST_H
//#define MAXWEAKCLASSIFIERSIZE 100


//int WeakClassifierBuffer[MAXWEAKCLASSIFIERSIZE];
//double WeakClassifierWeightBuffer[MAXWEAKCLASSIFIERSIZE];

typedef struct AdaBoostTrainData_ AdaBoostTrainData;
typedef struct AdaBoostClassifier_ AdaBoostClassifier;
typedef struct AdaBoostAtomClassifier_  AdaBoostAtomClassifier;
struct AdaBoostTrainData_ {
	unsigned int DataSize;
	int *data;
	char *label;
};
struct AdaBoostClassifier_ {
	int ClassifierSizeType0;
	int ClassifierSizeType1;
	int * ClassifierType0;
	int * ClassifierType1;//if data>=classifier&&data.label==1 type=1;else if data<classifier &&data.label==1 type=0
	double *ClassifierWeight0;
	double *ClassifierWeight1;
	
};
struct AdaBoostAtomClassifier_ {
	int Classifier;
	char type;
	double error;
};
AdaBoostClassifier AdaBoost(AdaBoostTrainData data,int size);
void ReleaseWeakClassifier(AdaBoostClassifier WeakClassifier);
#endif