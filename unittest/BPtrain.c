#include "fxBPNerveNetwork.h"

Fx_BPTrainDataSet * ReadTrainData(FILE *file){
	int traindata_size;
	int input_size;
	int output_size;
	fscanf(file, "%d %d %d\n", &traindata_size, &input_size, &output_size);
	Fx_BPTrainDataSet * traindata_set = (Fx_BPTrainDataSet *)fxMalloc(sizeof(Fx_BPTrainDataSet));
	traindata_set->size = traindata_size;
	traindata_set->traindata_array = (Fx_BPTrainData *)fxMalloc(sizeof(Fx_BPTrainData)*traindata_size);
	for (int j = 0; j < traindata_size; j++){
		
		
		(traindata_set->traindata_array)[j].input_array = (double*)fxMalloc(sizeof(double)*input_size);
		(traindata_set->traindata_array)[j].input_num = input_size;
		(traindata_set->traindata_array)[j].output_array = (double*)fxMalloc(sizeof(double)*output_size);
		(traindata_set->traindata_array)[j].output_num = output_size;
		(traindata_set->traindata_array)[j].residual_array = (double*)fxMalloc(sizeof(double)*output_size);
		(traindata_set->traindata_array)[j].residual_num = output_size;
		
		for (int i = 0; i < input_size; i++){
			fscanf(file, "%lf ", &(((traindata_set->traindata_array)[j].input_array)[i]));
		}
		for (int i = 0; i < output_size; i++){
			fscanf(file, "%lf ", &(((traindata_set->traindata_array)[j].output_array)[i]));
		}
		fscanf(file, "\n");
	}

	return traindata_set;

}



int main(){
	int nums_per_layer[4] = { 2 ,2,2,1 };
	Fx_BPNerveNetwork *network=fxCreateBP(nums_per_layer,4);
	fxCreateSynapseConnect(NULL, network,FX_TANH);
	FILE* file = fopen("D:\\DATA\\bp_data.txt","r");
	Fx_BPTrainDataSet *dataset = ReadTrainData(file);
	fclose(file);
	showData(dataset);
	fxTrainBP(dataset, network, dataset, 0.5, 1, FX_BP_STOCHASTIC);
	fxSaveBPNet("D:\\DATA\\BP\\", network);
	fxFreeBP(&network);
	
	return 0;
}