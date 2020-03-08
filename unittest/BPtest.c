#include "fxBPNerveNetwork.h"

Fx_BPTestDataSet * ReadTestData(FILE *file){
	int testdata_size;
	int input_size;
	int output_size;
	fscanf(file, "%d %d %d\n", &testdata_size, &input_size, &output_size);
	Fx_BPTestDataSet * testdata_set = (Fx_BPTestDataSet *)fxMalloc(sizeof(Fx_BPTrainDataSet));
	testdata_set->size = testdata_size;
	testdata_set->testdata_array = (Fx_BPTestData *)fxMalloc(sizeof(Fx_BPTrainData)*testdata_size);
	for (int j = 0; j < testdata_size; j++){
		
		
		(testdata_set->testdata_array)[j].input_array = (double*)fxMalloc(sizeof(double)*input_size);
		(testdata_set->testdata_array)[j].input_num = input_size;
		(testdata_set->testdata_array)[j].output_array = (double*)fxMalloc(sizeof(double)*output_size);
		(testdata_set->testdata_array)[j].output_num = output_size;
		
		for (int i = 0; i < input_size; i++){
			fscanf(file, "%lf ", &(((testdata_set->testdata_array)[j].input_array)[i]));
		}
		fscanf(file, "\n");
	}

	return testdata_set;

}



int main(){
	FILE *file = fopen("D:\\DATA\\bptest.bp", "r");
	Fx_BPTestDataSet *test_dataset = ReadTestData(file);
	Fx_BPNerveNetwork *network = FxReadBP("D:\\DATA\\BP\\");
	//showNet(network);
	for (int i = 0; i < test_dataset->size; i++){
		fxBP(&(test_dataset->testdata_array[i]), network);
	}
	fxSaveBPResult("D:\\DATA\\result.bp",test_dataset);
	fxFreeBP(&network);
	
	return 0;
}