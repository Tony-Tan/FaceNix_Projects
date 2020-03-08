#include "fa3000.h"
char* save_result_path = "D:\\Data\\FA3000\\51\\";
int main(){
	FaTrainData *traindata = faReadTrainData("D:\\Data\\51landmark\\face128\\", TRAIN_DATA_SIZE, save_result_path);
	faTrain(traindata, save_result_path);
	faSaveparam(save_result_path);
}