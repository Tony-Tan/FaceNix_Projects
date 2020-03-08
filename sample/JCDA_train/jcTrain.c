#include "jcTrain.h"
#include <time.h>
#define RANDOM_PERTURBATION_RANGE 8
char * weak_file_exe = ".fnx";
JcLandMark MEANSHAPE;
//初始化landmark
void * jcInitFaceShape(char *mean_shap_path,JcTrainData *train_data){
	
	int pos_num = train_data->positive_num;
	int neg_num = train_data->negative_num;
	int loop_i=0;
	char shap_name[100];
	sprintf(shap_name, "%s%s", mean_shap_path, "mean_shap.txt");
	FILE *file = fopen(shap_name, "w+");
	//计算平均位置
	for (int i = 0; i < LANDMARKTYPE; i++){
		double sum_x = 0;
		double sum_y = 0;
		for (loop_i = 0; loop_i < pos_num; loop_i++){
			sum_x += train_data->positive_sample[loop_i].GroundTruth->LandMark[i].x;
			sum_y += train_data->positive_sample[loop_i].GroundTruth->LandMark[i].y;
		}
		MEANSHAPE.LandMark[i].x = (double)sum_x / pos_num;
		MEANSHAPE.LandMark[i].y = (double)sum_y / pos_num;
		fprintf(file, "%lf %lf\n", MEANSHAPE.LandMark[i].x, MEANSHAPE.LandMark[i].y);
	}
	fclose(file);

	
//Init Positive Samples shape
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < pos_num; loop_i++){
		JcLandMark * ground_truth = train_data->positive_sample[loop_i].GroundTruth;
		JcLandMark * face_shape = train_data->positive_sample[loop_i].Face_Shap;
		JcLandMark * face_shape_delta = train_data->positive_sample[loop_i].Face_Shap_Delta;
		for (int i = 0; i < LANDMARKTYPE; i++){
			int perturbations_x = fxRandom(RANDOM_PERTURBATION_RANGE, FX_RANDOM_MEAN_0);
			int perturbations_y = fxRandom(RANDOM_PERTURBATION_RANGE, FX_RANDOM_MEAN_0);
			face_shape->LandMark[i].x = MEANSHAPE.LandMark[i].x + perturbations_x;
			face_shape->LandMark[i].y = MEANSHAPE.LandMark[i].y + perturbations_y;
			face_shape_delta->LandMark[i].x = ground_truth->LandMark[i].x - face_shape->LandMark[i].x;
			face_shape_delta->LandMark[i].y = ground_truth->LandMark[i].y - face_shape->LandMark[i].y;
		}
	}
//Init negative Samples shape
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < neg_num; loop_i++){
		JcLandMark * face_shape = train_data->negative_sample[loop_i].Face_Shap;
		JcLandMark * face_shape_delta = train_data->negative_sample[loop_i].Face_Shap_Delta;
		for (int i = 0; i < LANDMARKTYPE; i++){
			int perturbations_x = fxRandom(RANDOM_PERTURBATION_RANGE, FX_RANDOM_MEAN_0);
			int perturbations_y = fxRandom(RANDOM_PERTURBATION_RANGE, FX_RANDOM_MEAN_0);
			face_shape->LandMark[i].x = MEANSHAPE.LandMark[i].x + perturbations_x;
			face_shape->LandMark[i].y = MEANSHAPE.LandMark[i].y + perturbations_y;
			face_shape_delta->LandMark[i].x = 0.0;
			face_shape_delta->LandMark[i].y = 0.0;
		}
	}
}


//
//Train Progress
//
//
//
int jcRandNodeType(int cascade_stage){
	int Rou =(int)((double) (1 - 0.1*(cascade_stage+1)) * 100.0);
	int random = fxRandom(100,FX_RANDOM_MEAN_HALF_N);
	if (random < Rou){
		return DETECTION;
	}
	else{
		return ALIGNMENT;
	}
}





int jcLearnATree(JcTreeNode **root,JcTrainData *train_data,int cascade_stage,int tree_num,int tree_depth,int node_position){
	int facial_point = tree_num%LANDMARKTYPE;
	if (tree_depth >= TREE_DEPTH){
		*root = jcCreateTreeNode(LEAF);
		(*root)->feature_lf = fxMalloc(sizeof(JcLeafFeature));
		jcLearnLeafNode(train_data, node_position, facial_point, tree_num, (*root)->feature_lf);
		return 0;
	}
	else{
		int node_type = jcRandNodeType(cascade_stage);
		*root = jcCreateTreeNode(node_type);
		if (node_type == ALIGNMENT){
			(*root)->feature_al = (JcAlignmentFeature*)fxMalloc(sizeof(JcAlignmentFeature));
			jcLearnAlignmentNode(train_data, cascade_stage, facial_point, node_position, ((*root)->feature_al));
		}
		else if (node_type == DETECTION){
			(*root)->feature_dt = (JcDetectionFeature*)fxMalloc(sizeof(JcDetectionFeature));
			jcLearnDetectionNode(train_data, node_position, cascade_stage, ((*root)->feature_dt));

		}

		jcLearnATree(&((*root)->left_child), train_data, cascade_stage, tree_num, tree_depth + 1, jcTreeLeftChild(node_position));
		jcLearnATree(&((*root)->right_child), train_data, cascade_stage, tree_num, tree_depth + 1, jcTreeRightChild(node_position));

	}
	return 0;
}

void jcLearnWeak_CL_RE(JcTrainData *train_data, int cascade_stage,int tree_num, JcWeakCl_Re * weak_cl_re){

	weak_cl_re->landmark_num = tree_num%LANDMARKTYPE;
	jcLearnATree(&(weak_cl_re->root), train_data, cascade_stage, tree_num, 0, 0);// 0 depth & 0 num

}
/*******************************************************************************************/
// quick sort to find a threshold for precision-recall
void jcQuickSort(double * src_array,int left_num,int right_num){
	if (left_num >= right_num){
		return;
	}
	int i = left_num;
	int j = right_num;
	double key = src_array[left_num];

	while (i < j)                             {
		while (i < j && key <= src_array[j])
		{
			j--;
		}

		src_array[i] = src_array[j];

		while (i < j && key >= src_array[i]){
			i++;
		}

		src_array[j] = src_array[i];
	}

	src_array[i] = key;
	jcQuickSort(src_array, left_num, i - 1);
	jcQuickSort(src_array, i + 1, right_num);



}

double jcFindPosScore(JcTrainData * train_data, double recall_rate){
	printf("===================Finding score======================\n");
	double score = 0.0;
	int pos_num = train_data->positive_num;


	double * class_score_array = (double *)malloc(sizeof(double)*pos_num);
	for (int i = 0; i < pos_num; i++){
		class_score_array[i] = train_data->positive_sample[i].Class_Score;
	}
	jcQuickSort(class_score_array, 0, pos_num-1);
	score = class_score_array[(int)(pos_num*(1.0 - recall_rate))];

	int remove_neg_num = 0;
	for (int i = 0; i < train_data->negative_num; i++){
		if (train_data->negative_sample[i].Class_Score < score&&
			train_data->negative_sample[i].isAvailable)
			remove_neg_num++;
	
	}
	if (remove_neg_num / (pos_num*(1.0 - recall_rate)) < 10)
		score = class_score_array[0];


	free(class_score_array);
	printf("======================%10lf======================\n", score);
	printf("=======================================================\n");
	return score;

}
/*******************************************************************************************/




void jcRemoveNegaSamples(JcTrainData *train_data, double score){
	
	int neg_num = train_data->negative_num;
	int neg_available = 0;
	for (int i = 0; i < neg_num; i++){
		if (train_data->negative_sample[i].Class_Score < score&&
			train_data->negative_sample[i].isAvailable){
			train_data->negative_sample[i].isAvailable = 0;
			
			printf("Remove %d sample\n", i);
#ifdef DEBUG
			fprintf(logfile, "Remove %d sample\n", i);
#endif
		}
	}
}

int jcNegSampleRemoved(JcTrainData * train_data){
	int result = 0;
	for (int i = 0; i < train_data->negative_num; i++){
		if (train_data->negative_sample[i].isAvailable == 0){
			result++;
		}
	}
	return result;
}



void jcFreeCl_Re(JcWeakCl_Re ** Cascade,int size){
	for (int i = 0; i < size; i++){
		jcFreeTree((*Cascade)[i].root);
	}
	fxFree(Cascade);

}

/*************************************************************************************************/
//for debug to  show face shap


void jcSaveFaceShap(char *face_shape_path,JcTrainData * train_data){
	char file_name[128];
	int total_size = train_data->positive_num;
	for (int i = 0; i < total_size; i++){
		JcTrainSample *sample = &(train_data->positive_sample[i]);
		sprintf(file_name, "%s%d%s", face_shape_path, i, ".pts");
		FILE *file = fopen(file_name, "w+");
		for (int j = 0; j < LANDMARKTYPE; j++){
			fprintf(file, "%lf %lf\n", sample->Face_Shap_Delta->LandMark[j].x, sample->Face_Shap_Delta->LandMark[j].y);
		}
		fclose(file);
	}


}


/*************************************************************************************************/
void jcTrainJCDA(JcTrainData *train_data, char *forest_file_path){

	char JCDA_file_name[100];
	sprintf(JCDA_file_name, "%s%s%s", forest_file_path, "Cascade", weak_file_exe);
	FILE *file = fopen(JCDA_file_name, "w+");
	JcWeakCl_Re * Cascade = (JcWeakCl_Re*)fxMalloc(sizeof(JcWeakCl_Re)*TOTAL_WEAK_C_R);
	int weakcl_re_num = 0;
	for (int i = 0; (i < STAGES_OF_CASCADE)&&(weakcl_re_num<TOTAL_WEAK_C_R); i++){
		char JCDA_W_GLOBAL[512];
		sprintf(JCDA_W_GLOBAL, "%s%s%d%s", forest_file_path, "Cascade_Global_",i, weak_file_exe);
		//FILE *file_w = fopen(JCDA_W_GLOBAL, "w+");
		
		for (int j = 0; (j < WEAK_C_R_EACH_STAGE) && (weakcl_re_num<TOTAL_WEAK_C_R); j++, weakcl_re_num++){
			
			clock_t start, end;
			start = clock();
			printf("learning cascade stage:%d  weak a&d tree :%d   \n", i, j);
#ifdef DEBUG
			fprintf(logfile,"learning cascade stage:%d  weak a&d tree :%d   \n", i, j);
#endif
			jcUpdateSampleWeight(train_data);
			jcLearnWeak_CL_RE(train_data, i, j, &(Cascade[weakcl_re_num]));
#define RECALL_RATE 0.95
			double score = jcFindPosScore(train_data, RECALL_RATE);
			
			Cascade[weakcl_re_num].class_threshold = score;
			Cascade[weakcl_re_num].landmark_num = j%LANDMARKTYPE;
			//jcRemoveNegaSamples(train_data, score);
			jcRemoveNegaSamples(train_data, score);
			int neg_removed = jcNegSampleRemoved(train_data);
#define MING_RATE 0.1
			if (((double)neg_removed / (double)(train_data->negative_num))>=MING_RATE)
				jcMiningNegSample(train_data, Cascade, weakcl_re_num );
			jcSaveCl_ReFile(file, &Cascade[weakcl_re_num]);
			/**************************************************************************************************/
			end = clock();
			double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000;
			printf("|Mission Complete %lf %% using time:%lf ms\n", (double)((weakcl_re_num + 1.0) / TOTAL_WEAK_C_R*100.0), ms);
			printf("ETA:%lf h\n", ms*(TOTAL_WEAK_C_R - (weakcl_re_num + 1.0)) / (1000.0 * 60 * 60));
			
		}
		jcGlobal(train_data, i, JCDA_W_GLOBAL);
		
		jcUpdateShape(train_data, i);
	}
	fclose(file);
	jcFreeCl_Re(&Cascade, WEAK_C_R_EACH_STAGE);

}

void jcSaveTree(FILE*file, JcTreeNode *root){
	switch (root->node_type)
	{
	case DETECTION:
	{
		fprintf(file, "D %d %d %d %d %d %d %d\n", root->feature_dt->FacialPoint[0], root->feature_dt->FacialPoint[1],
				root->feature_dt->off_set[0].x, root->feature_dt->off_set[0].y,
				root->feature_dt->off_set[1].x, root->feature_dt->off_set[1].y,
				root->feature_dt->threshold);
		jcSaveTree(file, root->left_child);
		jcSaveTree(file, root->right_child);
		break;
	}
	case ALIGNMENT:
	{
		fprintf(file, "A %d %d %d %d %d\n", root->feature_al->off_set[0].x, root->feature_al->off_set[0].y,
				root->feature_al->off_set[1].x, root->feature_al->off_set[1].y,
				root->feature_al->threshold);
		jcSaveTree(file, root->left_child);
		jcSaveTree(file, root->right_child);
		break;
	}
	case LEAF:
	{
		fprintf(file, "L %lf\n", root->feature_lf->class_score);
		break;
	}
	default:
		break;
	}


}

void jcSaveCl_ReFile(FILE * file, JcWeakCl_Re * weak){
	fprintf(file, "C %d %lf\n", weak->landmark_num, weak->class_threshold);
	jcSaveTree(file, weak->root);
	printf("\n\n\n***********************************************************\n");
	printf("Cascade File Saved !\n");
	printf("***********************************************************\n\n\n");
#ifdef DEBUG
	fprintf(logfile,"\n***********************************************************\n");
	fprintf(logfile, "Cascade File Saved !\n");
	fprintf(logfile, "***********************************************************\n");
#endif
}