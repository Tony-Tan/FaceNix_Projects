#include "faGlobal.h"
static void faCreateProblem_x(struct feature_node*** x,int l,int n){
	*x = (struct feature_node**)malloc(sizeof(struct feature_node*)*l);
	for (int i = 0; i < l; i++){
		(*x)[i] = (struct feature_node*)malloc(sizeof(struct feature_node)*n);
	}

}
static void faFreeProblem_x(struct feature_node** x, int l){
	for (int i = 0; i < l; i++){
		free(x[i]);
	}
	free(x);
}


static double w[LANDMARK_TYPE * 2][(1 << TREE_DEPTH) * TREE_NUM_EACH_STAGE];
void faGlobal(FaTrainData *train_data,char* global_file_nam){
	//set problem
	struct problem problem;
	problem.l = TRAIN_DATA_SIZE ;
	problem.n = (1 << TREE_DEPTH) * TREE_NUM_EACH_STAGE;
	
	problem.bias = -1;
	
	//set param
	struct parameter  param;
	param.C = 1.0/ TRAIN_DATA_SIZE;
	param.eps = 0.1;
	param.p = 0;
	param.solver_type = L2R_L2LOSS_SVR_DUAL;
	param.init_sol = NULL;
	
	//malloc x 
	faCreateProblem_x(&(problem.x), problem.l,  TREE_NUM_EACH_STAGE + 1);
	//malloc y
	problem.y = (double *)malloc(sizeof(double) * problem.l);


	//file create

	FILE *file = fopen(global_file_nam, "w+");
	fprintf(file, "%d %d\n", LANDMARK_TYPE * 2, problem.n);
	
	for (int i = 0; i < TRAIN_DATA_SIZE; i++){
		for (int j = 0; j < TREE_NUM_EACH_STAGE; j++){
			problem.x[i][j].index = train_data->data_array[i].lbf[j] + j*(1 << TREE_DEPTH) + 1;
			problem.x[i][j].value = 1.0;

		}
		problem.x[i][TREE_NUM_EACH_STAGE].index = -1;
	}

	//train:
	for (int l = 0; l < LANDMARK_TYPE * 2; l++){
		int L = l / 2;
		if (! (l % 2)){
			//regression x
			for (int i = 0; i < TRAIN_DATA_SIZE; i++){
				problem.y[i] = (double)train_data->data_array[i].landmark_delta[L].x;
			}
			if (NULL == check_parameter(&problem, &param)){
				struct model* model = train(&problem, &param);
				for (int w_size = 0; w_size < problem.n; w_size++){
					fprintf(file, "%lf ", model->w[w_size]);
					w[l][w_size] = model->w[w_size];
				}
				fprintf(file, "\n");
				free_and_destroy_model(&model);
			}


		}
		else{
			//regression y
			for (int i = 0; i < TRAIN_DATA_SIZE; i++){
				problem.y[i] = (double)train_data->data_array[i].landmark_delta[L].y;
			}
			//
			//
			//
			if (NULL == check_parameter(&problem, &param)){
				struct model* model = train(&problem, &param);
				for (int w_size = 0; w_size < problem.n; w_size++){
					fprintf(file, "%lf ", model->w[w_size]);
					w[l][w_size] = model->w[w_size];
				}
				fprintf(file, "\n");
				free_and_destroy_model(&model);
			}

		}
	
	}
	fclose(file);
	faFreeProblem_x(problem.x, problem.l);
	free(problem.y);

}

void faUpdateShape(FaTrainData *train_data){
	int loop_i = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads( OPENMP_THREAD_NUM )
#endif
	for (loop_i = 0; loop_i < train_data->data_size; loop_i++){
		for (int l = 0; l < LANDMARK_TYPE; l++){
			for (int j = 0; j < TREE_NUM_EACH_STAGE; j++){
				FaData *data_sample = &(train_data->data_array[loop_i]);
				int w_num = data_sample->lbf[j] + j*(1 << TREE_DEPTH);
				data_sample->landmark_realtime[l].x = data_sample->landmark_realtime[l].x + w[2*l][w_num];
				data_sample->landmark_realtime[l].y = data_sample->landmark_realtime[l].y + w[2*l+1][w_num];
				data_sample->landmark_delta[l].x = data_sample->landmark_tar[l].x - data_sample->landmark_realtime[l].x;
				data_sample->landmark_delta[l].y = data_sample->landmark_tar[l].y - data_sample->landmark_realtime[l].y;
			}
		}
	}

}