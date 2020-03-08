#include "jcNegSampleMining.h"


char * NegData_Back_Path = "non_face\\"; 



JcRemovedSampleSet *jcGetRemovedNegSample(JcTrainData *train_data){
	int neg_size = train_data->negative_num;
	int set_size = 0;
	int loop_i = 0;
	JcTrainSample *neg_sample = train_data->negative_sample;
#ifdef _OPENMP
#pragma omp parallel for reduction (+ : set_size)
#endif
	for (loop_i = 0; loop_i < neg_size; loop_i++){
		if (!neg_sample[loop_i].isAvailable)
			set_size++;
	}
	JcRemovedSampleSet *removed_set = (JcRemovedSampleSet *)fxMalloc(sizeof(JcRemovedSampleSet));
	removed_set->removed_size = set_size;
	removed_set->removed_position = (int *)fxMalloc(sizeof(int)*set_size);
	for (int i = 0, j = 0; i < neg_size; i++){
		if (!neg_sample[i].isAvailable){
			removed_set->removed_position[j] = i;
			j++;
		}
	}
	return removed_set;

}
void jcFreeRemovedNeg_Sample(JcRemovedSampleSet ** sampleset){
	if ((*sampleset)->removed_size!=0)
		fxFree(&(*sampleset)->removed_position);
	fxFree(sampleset);
}


int jcTestSample(JcTrainSample *sample, JcWeakCl_Re * cl_re, int size){

	for (int i = 0; i < STAGES_OF_CASCADE; i++){
		for (int j = 0; j < WEAK_C_R_EACH_STAGE ; j++){
			if ((i*WEAK_C_R_EACH_STAGE + j) > size){
				return 1;
			}

			JcWeakCl_Re * weakcl_re = &(cl_re[j + i*WEAK_C_R_EACH_STAGE]);
			JcTreeNode *tree = weakcl_re->root;
			int al_facial_point = weakcl_re->landmark_num;
			double treshold = weakcl_re->class_threshold;
			int position = 0;
			while (tree->node_type != LEAF){
				if (tree->node_type == DETECTION){
					int *face_point = tree->feature_dt->FacialPoint;
					FxPoint center[2];
					center[0].x = (int)sample->Face_Shap->LandMark[face_point[0]].x;
					center[0].y = (int)sample->Face_Shap->LandMark[face_point[0]].y;
					center[1].x = (int)sample->Face_Shap->LandMark[face_point[1]].x;
					center[1].y = (int)sample->Face_Shap->LandMark[face_point[1]].y;
					int value=jcDetectionFeatureValue(sample->mat, center, tree->feature_dt->off_set);
					if (value < tree->feature_dt->threshold){
						tree = tree->left_child;
						position = position * 2 + 1;
					}
					else{
						tree = tree->right_child;
						position = position * 2 + 2;
					}
				}
				else if (tree->node_type == ALIGNMENT){
					int value = jcAlignmentFeatureValue(sample->mat, &(sample->Face_Shap->LandMark[al_facial_point]), tree->feature_al->off_set);
					if (value < tree->feature_al->threshold){
						tree = tree->left_child;
						position = position * 2 + 1;
					}
					else{
						tree = tree->right_child;
						position = position * 2 + 2;
					}
				}
				


			}
			sample->LBF[j] = position-(1<<TREE_DEPTH)+1+j*(1<<TREE_DEPTH);
			sample->Class_Score += tree->feature_lf->class_score;
			if (sample->Class_Score < treshold)
				return 0;
		}
		//
		if (size / WEAK_C_R_EACH_STAGE > i)
			jcUpdateSampleShape(sample, i,1);
	}
	return 1;
}
#define SCALE_MAX 100
int BIGNEG_WIDTH = 0;
int BIGNEG_HEIGHT = 0;
FxMat * BIGNEG_MAT=NULL;
int BIGNEG_NUM = 0;
int BIGNEG_SCALE = 0;
char *BIGNEG_PATH = "D:\\Faces_Database\\non-face-jpg\\";
char * exe = ".jpg";
FxMat * jcSegBigNegSample(){
	
	if (BIGNEG_MAT == NULL){
		char neg_back_name[100];
		sprintf(neg_back_name, "%s%d%s", BIGNEG_PATH, BIGNEG_NUM, exe);
		BIGNEG_MAT = readImage(neg_back_name, 0);
	}
	else if (BIGNEG_MAT->height <= BIGNEG_HEIGHT + SAMPLE_HEIGHT){
		fxReleaseMat(&BIGNEG_MAT);
		if (BIGNEG_NUM<TOTALNEGSIZE-1)
			BIGNEG_NUM++;
		else{
			BIGNEG_SCALE++;
			BIGNEG_NUM = 0;
		}
		char neg_back_name[100];
		sprintf(neg_back_name, "%s%d%s", BIGNEG_PATH, BIGNEG_NUM, exe);
		BIGNEG_MAT = readImage(neg_back_name, 0);
		if (BIGNEG_SCALE != 0){
			FxMat* temp = fxCreateMat(fxSize(BIGNEG_MAT->width * (SCALE_MAX - BIGNEG_SCALE) / SCALE_MAX, BIGNEG_MAT->height*(SCALE_MAX - BIGNEG_SCALE) / SCALE_MAX), BIGNEG_MAT->type);
			fxResize(BIGNEG_MAT,temp,FX_INTER_LINEAR);
			fxReleaseMat(&BIGNEG_MAT);
			BIGNEG_MAT = temp;
		}
		BIGNEG_WIDTH = 0;
		BIGNEG_HEIGHT = 0;

	}
	FxMat * smale_neg = fxCreateMat(fxSize(SAMPLE_WIDTH, SAMPLE_HEIGHT), FX_8C1);
	int width = BIGNEG_MAT->width;
	uchar * smale_neg_data = smale_neg->data;
	uchar * big_neg_data = BIGNEG_MAT->data + (BIGNEG_HEIGHT*width + BIGNEG_WIDTH);
	for (int i = 0; i < SAMPLE_HEIGHT; i++){
		memcpy(smale_neg_data, big_neg_data, SAMPLE_WIDTH*sizeof(uchar));
		smale_neg_data += smale_neg->width;
		big_neg_data += width;
	}
	
	BIGNEG_WIDTH ++;
	if (BIGNEG_WIDTH + SAMPLE_WIDTH >= BIGNEG_MAT->width){
		BIGNEG_HEIGHT ++;
		BIGNEG_WIDTH = 0;
	}
	//printf("Mining: %d Negative Pic width :%d  height %d  \n", BIGNEG_NUM, BIGNEG_WIDTH, BIGNEG_HEIGHT);
	return smale_neg;
}



void jcMiningNegSample(JcTrainData *train_data, JcWeakCl_Re * cl_re, int size){
	JcRemovedSampleSet * removed_sample_set = jcGetRemovedNegSample(train_data);
	
	
	for (int  j = 0; j<removed_sample_set->removed_size;){ 
		int neg_sampel2remove_position = removed_sample_set->removed_position[j];
		JcTrainSample neg_sample;
		memset(&(neg_sample.LBF), 0, sizeof(int)*WEAK_C_R_EACH_STAGE);
		//sprintf(neg_back_name, "%s%s%d%s", TrainData_Path, NegData_Back_Path, i, ".png");
		neg_sample.Class_Score = 0.0; 
		neg_sample.Sample_Weight = 1.0;
		//to free
		neg_sample.Face_Shap = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		neg_sample.Face_Shap_Delta = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
		for (int j = 0; j < LANDMARKTYPE; j++){
			neg_sample.Face_Shap->LandMark[j].x = MEANSHAPE.LandMark[j].x;
			neg_sample.Face_Shap->LandMark[j].y = MEANSHAPE.LandMark[j].y;
			neg_sample.Face_Shap_Delta->LandMark[j].x =0;
			neg_sample.Face_Shap_Delta->LandMark[j].y =0;
		
		}
		//to free 
		neg_sample.mat = jcSegBigNegSample();
		if (jcTestSample(&neg_sample, cl_re, size)){
			JcTrainSample * neg_sample_temp = &(train_data->negative_sample[neg_sampel2remove_position]);
			fxFree(&(neg_sample_temp->Face_Shap));
			fxReleaseMat(&(neg_sample_temp->mat));

			neg_sample_temp->mat = neg_sample.mat;
			neg_sample_temp->Class_Score = neg_sample.Class_Score;
			neg_sample_temp->GroundTruth = NULL;
			neg_sample_temp->Face_Shap = neg_sample.Face_Shap;
			neg_sample_temp->Face_Shap_Delta = neg_sample.Face_Shap_Delta;
			neg_sample_temp->isAvailable = 1;
			neg_sample_temp->Label = -1;
			neg_sample_temp->Position_in_Tree = 0;
			neg_sample_temp->Sample_Weight = exp(neg_sample.Class_Score);
			memcpy(neg_sample_temp->LBF, neg_sample.LBF, sizeof(int)*WEAK_C_R_EACH_STAGE);
			j++;
			printf("-----------------------------Negative mining-----------------------------\n");
			printf("%d replaced by NO. %d SCALE %d non_face_back width: %d height: %d\n", neg_sampel2remove_position, BIGNEG_NUM, BIGNEG_SCALE, BIGNEG_WIDTH, BIGNEG_HEIGHT);
			printf("-------------------------------------------------------------------------\n");
#ifdef DEBUG
			fprintf(logfile,"--------------------Negative mining-----------------------------\n");
			fprintf(logfile, "%d replaced by NO. %d SCALE %d non_face_back width: %d height: %d\n", neg_sampel2remove_position, BIGNEG_NUM, BIGNEG_SCALE, BIGNEG_WIDTH, BIGNEG_HEIGHT);
			fprintf(logfile,"----------------------------------------------------------------\n");
#endif
		}
		else{
			fxFree(&(neg_sample.Face_Shap));
			fxFree(&(neg_sample.Face_Shap_Delta));
			fxReleaseMat(&(neg_sample.mat));
		
		}
	}

	jcFreeRemovedNeg_Sample(&removed_sample_set);
}