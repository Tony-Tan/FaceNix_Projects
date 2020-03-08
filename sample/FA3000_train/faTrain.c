#include "faTrain.h"



void faInitTarinData(FaTrainData *traindata){
	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(OPENMP_THREAD_NUM)
#endif
	for (loop_i = 0; loop_i < traindata->data_size; loop_i++){
		traindata->data_array[loop_i].tree_position = 0;
	
	}



}
void faTrain(FaTrainData *traindata, char * result_path){


	char local_file_name[512];
	char global_file_name[512];
	for (int i = 0; i < STAGE; i++){
		faInitLBF(traindata);
		sprintf(local_file_name, "%s%s%d%s", result_path, "stage_", i, "_l.fa");
		sprintf(global_file_name, "%s%s%d%s", result_path, "stage_", i, "_g.fa");
		FILE *file = fopen(local_file_name, "w+");
		for (int j = 0; j < TREE_NUM_EACH_STAGE; j++){
			
			int landmark_num = j % LANDMARK_TYPE;
			faInitTarinData(traindata);
			printf("----------------------------------------------------\n");
			printf("%d:\n", landmark_num);
			printf("Stage:%d  No.%d\n", i, j);
			printf("Tree learning...\n");
			printf("----------------------------------------------------\n\n");
			FaTreeNode *root;

			faLocalTree(&root, 0, traindata, landmark_num, 0, i);
			faLBF(traindata, j);
			faSaveTree(file, root, landmark_num);
			faFreeTree(&root);
			
		}
		fclose(file);
		faGlobal(traindata, global_file_name);
		faUpdateShape(traindata);
		//printf("Data1 LBF:\n");
		//for (int k = 0; k < traindata->data_size; k++){
		//	for (int j = 0; j < TREE_NUM_EACH_STAGE; j++){
		//		printf("%d ", traindata->data_array[k].lbf[j]);
		//	}
		//}
		//printf("\n===========================LBF============================\n");
	}


}