#include "jcGlobal.h"
static void jcCreateProblem_x(struct feature_node*** x,int l,int n){
	*x = (struct feature_node**)malloc(sizeof(struct feature_node*)*l);
	for (int i = 0; i < l; i++){
		(*x)[i] = (struct feature_node*)malloc(sizeof(struct feature_node)*n);
	}

}
static void jcFreeProblem_x(struct feature_node** x, int l){
	for (int i = 0; i < l; i++){
		free(x[i]);
	}
	free(x);
}


static double w[STAGES_OF_CASCADE][LANDMARKTYPE * 2][(1 << TREE_DEPTH) * WEAK_C_R_EACH_STAGE];
void jcGlobal(JcTrainData *train_data,int stage_num, char* global_file_nam){
	//set problem
	struct problem problem;
	problem.l = train_data->positive_num ;
	problem.n = (1 << TREE_DEPTH) * WEAK_C_R_EACH_STAGE;
	
	problem.bias = -1;
	
	//set param
	struct parameter  param;
	param.C = 1;
	param.eps = 0.1;
	param.nr_weight = 0;
	param.p = 0.001;
	param.solver_type = L2R_L2LOSS_SVR;
	param.init_sol = NULL;
	
	//malloc x 
	jcCreateProblem_x(&(problem.x), problem.l, WEAK_C_R_EACH_STAGE + 1);
	//malloc y
	problem.y = (double *)malloc(sizeof(double) * problem.l);


	//file create

	FILE *file = fopen(global_file_nam, "w+");
	fprintf(file, "%d %d\n", LANDMARKTYPE * 2, problem.n);
	
	for (int i = 0; i < train_data->positive_num; i++){
		for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
			problem.x[i][j].index = train_data->positive_sample[i].LBF[j]  + 1;
			problem.x[i][j].value = 1.0;

		}
		problem.x[i][WEAK_C_R_EACH_STAGE].index = -1;
	}

	//train:
	for (int l = 0; l < LANDMARKTYPE * 2; l++){
		int L = l / 2;
		if (! (l % 2)){
			//regression x
			for (int i = 0; i < train_data->positive_num; i++){
				problem.y[i] = (double)train_data->positive_sample[i].Face_Shap_Delta->LandMark[L].x;
			}
			if (NULL == check_parameter(&problem, &param)){
				struct model* model = train(&problem, &param);
				for (int w_size = 0; w_size < problem.n; w_size++){
					fprintf(file, "%lf ", model->w[w_size]);
					w[stage_num][l][w_size] = model->w[w_size];
				}
				fprintf(file, "\n");
				free_and_destroy_model(&model);
			}


		}
		else{
			//regression y
			for (int i = 0; i < train_data->positive_num; i++){
				problem.y[i] = (double)train_data->positive_sample[i].Face_Shap_Delta->LandMark[L].y;
			}
			//
			//
			//
			if (NULL == check_parameter(&problem, &param)){
				struct model* model = train(&problem, &param);
				for (int w_size = 0; w_size < problem.n; w_size++){
					fprintf(file, "%lf ", model->w[w_size]);
					w[stage_num][l][w_size] = model->w[w_size];
				}
				fprintf(file, "\n");
				free_and_destroy_model(&model);
			}

		}
	
	}
	fclose(file);
	jcFreeProblem_x(problem.x, problem.l);
	free(problem.y);

}

void jcUpdateShape(JcTrainData *train_data,int stage_num){
	int loop_i = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads( OPENMP_THREAD_NUM )
#endif
	for (loop_i = 0; loop_i < train_data->positive_num; loop_i++){
		for (int l = 0; l < LANDMARKTYPE; l++){
			for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
				JcTrainSample *data_sample = &(train_data->positive_sample[loop_i]);
				jcUpdateSampleShape(data_sample, stage_num,0);
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads( OPENMP_THREAD_NUM )
#endif
	for (loop_i = 0; loop_i < train_data->negative_num; loop_i++){
		for (int l = 0; l < LANDMARKTYPE; l++){
			for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
				
				JcTrainSample *data_sample = &(train_data->negative_sample[loop_i]);
				jcUpdateSampleShape(data_sample, stage_num,1);
				
			}
		}
	}
}


void jcUpdateSampleShape(JcTrainSample *data_sample, int stage_num,int isNeg){
	for (int l = 0; l < LANDMARKTYPE; l++){
		for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
			if (data_sample->isAvailable){
				int w_num = data_sample->LBF[j];
				data_sample->Face_Shap->LandMark[l].x = data_sample->Face_Shap->LandMark[l].x + w[stage_num][2 * l][w_num];
				data_sample->Face_Shap->LandMark[l].y = data_sample->Face_Shap->LandMark[l].y + w[stage_num][2 * l + 1][w_num];
				if (!isNeg){
					data_sample->Face_Shap_Delta->LandMark[l].x = data_sample->GroundTruth->LandMark[l].x - data_sample->Face_Shap->LandMark[l].x;
					data_sample->Face_Shap_Delta->LandMark[l].y = data_sample->GroundTruth->LandMark[l].y - data_sample->Face_Shap->LandMark[l].y;
				}
			}
		}
	}
}