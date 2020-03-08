#include "jcda.h"
char * PosData_Path = "face\\";
char * NegData_Path = "non_face\\";

//分配训练数据内存空间
JcTrainData* jcCreateTrainData(int Positive_Num, int Negative_Num){
	JcTrainData * train_data = (JcTrainData*)fxMalloc(sizeof(JcTrainData));
	train_data->negative_sample = (JcTrainSample*)fxMalloc(sizeof(JcTrainSample)*Negative_Num);
	train_data->negative_num = Negative_Num;
	train_data->positive_sample = (JcTrainSample*)fxMalloc(sizeof(JcTrainSample)*Positive_Num);
	train_data-> positive_num = Positive_Num;
	return train_data;
}
//读取landmark信息
void jcReadPTS(char* pts_name,JcLandMark * landmark){

	FILE * file_name = fopen(pts_name, "r");
	int loop_i = 0;
	for (loop_i = 0; loop_i < LANDMARKTYPE; loop_i++){
		double x, y;
		fscanf(file_name, "%lf %lf\n", &x, &y);
		landmark->LandMark[loop_i].x = x;
		landmark->LandMark[loop_i].y = y;
	}
	fclose(file_name);

}
//读取训练样本数据
void jcReadTrainData(JcTrainData* TrainData, char * TrainDataPath){
	int positive_num = TrainData->positive_num;
	int negative_num = TrainData->negative_num;

	JcTrainSample * pos_sample_array = TrainData->positive_sample;
	JcTrainSample * neg_sample_array = TrainData->negative_sample;
	int loop_i = 0;
	int progress_i=0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(4)
#endif
	for (loop_i = 0; loop_i < positive_num; loop_i++){
		char pos_image_name[100];
		char pos_pts_name[100];
		sprintf(pos_image_name, "%s%s%d%s", TrainDataPath, PosData_Path, loop_i, ".png");
		sprintf(pos_pts_name, "%s%s%d%s", TrainDataPath, PosData_Path, loop_i, ".pts");
		pos_sample_array[loop_i].Class_Score = 0.0;
		pos_sample_array[loop_i].Sample_Weight = 1.0;
		pos_sample_array[loop_i].Label = POSITIVE;
		pos_sample_array[loop_i].GroundTruth = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		pos_sample_array[loop_i].Face_Shap = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		pos_sample_array[loop_i].Face_Shap_Delta = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		pos_sample_array[loop_i].mat = readImage(pos_image_name, 0);
		pos_sample_array[loop_i].Position_in_Tree = 0;
		pos_sample_array[loop_i].isAvailable = 1;
		jcReadPTS(pos_pts_name, pos_sample_array[loop_i].GroundTruth);
#ifdef _OPENMP
#pragma omp critical
#endif
		progress_i++;
		fxProgressBar("Reading Pos_Sample", progress_i-1, positive_num);
	}

	//Negative Samples
	progress_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(4)
#endif
	for (loop_i = 0; loop_i < negative_num; loop_i++){
		char neg_image_name[100];
		sprintf(neg_image_name, "%s%s%d%s", TrainDataPath, NegData_Path, loop_i, ".png");
		neg_sample_array[loop_i].Class_Score = 0.0;
		neg_sample_array[loop_i].Sample_Weight = 1.0;
		neg_sample_array[loop_i].Label = NEGATIVE;
		neg_sample_array[loop_i].GroundTruth = NULL;
		neg_sample_array[loop_i].Face_Shap = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		neg_sample_array[loop_i].Face_Shap_Delta = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		neg_sample_array[loop_i].mat = readImage(neg_image_name, 0);
		neg_sample_array[loop_i].Position_in_Tree = 0;
		neg_sample_array[loop_i].isAvailable = 1;
#ifdef _OPENMP
#pragma omp critical
#endif
		progress_i++;
		fxProgressBar("Reading Neg_Sample", progress_i-1, negative_num);
	}


}
//释放分配的训练样本空间
void jcFreeTrainData(JcTrainData** TrainData){
	int neg_num = (*TrainData)->negative_num;
	int pos_num = (*TrainData)->positive_num;
	
	JcTrainSample* pos_sample = (*TrainData)->positive_sample;
	JcTrainSample* neg_sample = (*TrainData)->negative_sample;


	int loop_i = 0;
	for (loop_i = 0; loop_i < pos_num; loop_i++){
		fxFree(&(pos_sample[loop_i].GroundTruth));
		fxFree(&(pos_sample[loop_i].Face_Shap));
		fxFree(&(pos_sample[loop_i].Face_Shap_Delta));
		fxReleaseMat(&(pos_sample[loop_i].mat));
		fxProgressBar("Free Pos_Sample", loop_i, pos_num);
	}
	fxFree(&((*TrainData)->positive_sample));

	for (loop_i = 0; loop_i < neg_num; loop_i++){
		fxFree(&(neg_sample[loop_i].Face_Shap));
		fxFree(&(neg_sample[loop_i].Face_Shap_Delta));
		fxReleaseMat(&(neg_sample[loop_i].mat));
		fxProgressBar("Free Neg_Sample", loop_i, neg_num);
	}
	fxFree(&((*TrainData)->negative_sample));
}