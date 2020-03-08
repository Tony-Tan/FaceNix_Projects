#include "jcda.h"
#include "jcTrain.h"
#include "jcNegSampleMining.h"
#define POS_SAMPLE_COUNT 1600
#define NEG_SAMPLE_COUNT 1600

char * TrainData_Path = "D:\\Faces_Database\\FaceNixData80\\";
char * Cascade_path = "D:\\DATA\\cascade80\\";
char * Debub_log = "D:\\DATA\\cascade80\\log.txt";
FILE * logfile=NULL;
int main(){
#ifdef DEBUG
	logfile = fopen(Debub_log, "w+");
#endif
	JcTrainData * trainData = jcCreateTrainData(POS_SAMPLE_COUNT,NEG_SAMPLE_COUNT);
	jcReadTrainData(trainData, TrainData_Path);
	jcInitFaceShape("D:\\DATA\\cascade80\\",trainData);
	///
	jcTrainJCDA(trainData, Cascade_path);
	///
	jcFreeTrainData(&trainData);
#ifdef DEBUG
	fclose(logfile);
#endif
}