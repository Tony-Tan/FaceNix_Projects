#include "faLBF.h"
void faLBF(FaTrainData *train_data,int LBF_num){
	int loop_i = 0; 
#ifdef _OPENMP
#	pragma omp parallel for num_threads(OPENMP_THREAD_NUM)
#endif
	for (loop_i = 0; loop_i < train_data->data_size; loop_i++){
		train_data->data_array[loop_i].lbf[LBF_num] = train_data->data_array[loop_i].tree_position - (1 << TREE_DEPTH) + 1;
	}
	printf("LBF create done!\n");
	printf("***************************************************************************\n\n\n");
}

void faInitLBF(FaTrainData *train_data){
	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(OPENMP_THREAD_NUM)
#endif
	for (loop_i = 0; loop_i < train_data->data_size; loop_i++){
		for (int i = 0; i < TREE_NUM_EACH_STAGE ; i++)
		train_data->data_array[loop_i].lbf[i] = 0;
	}
	printf("LBF create done!\n");
	printf("***************************************************************************\n\n\n");
}
