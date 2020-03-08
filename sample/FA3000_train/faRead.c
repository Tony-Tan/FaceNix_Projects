#include "faRead.h"
#define PTS_EX ".pts"
#define IMG_EX ".png"
FaTrainData * faCreateTrainData(int data_size){
	FaTrainData *train_data = (FaTrainData*)fxMalloc(sizeof(FaTrainData));
	train_data->data_size = data_size;
	train_data->data_array = (FaData*)fxMalloc(sizeof(FaData)*data_size);
	return train_data;
}
void faFreeTrainData(FaTrainData **traindata){
	
	for (int i = 0; i < (*traindata)->data_size; i++){
		fxReleaseMat(&((*traindata)->data_array[i].image));
	}
	fxFree(&((*traindata)->data_array));
	fxFree(traindata);
}

FaTrainData * faReadTrainData(char * traindata_path, int traindata_size, char *mean_shape_path){

	//create train data
	FaTrainData* traindata = faCreateTrainData(traindata_size);

	int loop_i = 0, progress_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(OPENMP_THREAD_NUM)
#endif
	for (loop_i = 0; loop_i < traindata_size; loop_i++){
		char data_name[512];
		//read image
		sprintf(data_name, "%s%d%s", traindata_path, loop_i, IMG_EX);
		traindata->data_array[loop_i].image = readImage(data_name, 0);
		//read landmark
		sprintf(data_name, "%s%d%s", traindata_path, loop_i, PTS_EX);
		FILE* file = fopen(data_name, "r");
		double temp_x, temp_y;
		for (int i = 0; i < LANDMARK_TYPE; i++){
			fscanf(file,"%lf %lf\n", &temp_x, &temp_y);
			traindata->data_array[loop_i].landmark_tar[i].x = temp_x;
			traindata->data_array[loop_i].landmark_tar[i].y = temp_y;
		}
		traindata->data_array[loop_i].tree_position = 0;
		fclose(file);
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			progress_i++;
			fxProgressBar("Reading Sample", progress_i - 1, traindata_size); 
		
		}
	}
	faDeltaLandmark(traindata, mean_shape_path);
	return traindata;
}

void faDeltaLandmark(FaTrainData* traindata,char * mean_shape_path){

	FxPoint64 temp_sum[LANDMARK_TYPE];
	memset(temp_sum, 0, sizeof(FxPoint64)*LANDMARK_TYPE);
	
	for (int i = 0; i < traindata->data_size; i++){
		for (int j = 0; j < LANDMARK_TYPE; j++){
			temp_sum[j].x += traindata->data_array[i].landmark_tar[j].x;
			temp_sum[j].y += traindata->data_array[i].landmark_tar[j].y;
		}
	}
	char mean_shape_file[512];
	sprintf(mean_shape_file, "%smean_shape%s", mean_shape_path, PTS_EX);
	FILE* file = fopen(mean_shape_file,"w+");
	for (int j = 0; j < LANDMARK_TYPE; j++){
		temp_sum[j].x /= (double)traindata->data_size;
		temp_sum[j].y /= (double)traindata->data_size;
		fprintf(file, "%lf %lf\n", temp_sum[j].x, temp_sum[j].y);
	}
	fclose(file);
	for (int i = 0; i < traindata->data_size; i++){
		for (int j = 0; j < LANDMARK_TYPE; j++){
			traindata->data_array[i].landmark_realtime[j].x = temp_sum[j].x + fxRandom(INIT_MEAN_SHAPE_NOICE_RADIUS, FX_RANDOM_MEAN_0);
			traindata->data_array[i].landmark_realtime[j].y = temp_sum[j].y + fxRandom(INIT_MEAN_SHAPE_NOICE_RADIUS, FX_RANDOM_MEAN_0);
			traindata->data_array[i].landmark_delta[j].x = traindata->data_array[i].landmark_tar[j].x - traindata->data_array[i].landmark_realtime[j].x;
			traindata->data_array[i].landmark_delta[j].y = traindata->data_array[i].landmark_tar[j].y - traindata->data_array[i].landmark_realtime[j].y;
		}
	}

}