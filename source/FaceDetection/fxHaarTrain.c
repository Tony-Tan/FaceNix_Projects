#include"AdaBoost.h"
#include"Haar.h"
#include <float.h>
#include <time.h>
//#include <omp.h>
#define FX_ADABOOST_CLASSIFIER_SIZE 100
#define FX_MIN_TRUE_NEGATIVE_RATE 0.3
#define FX_CASCADE_STAGE_THRESHOLD 0.4
#define FX_CASCADE_MAX_STAGE 30


void fxReleaseHaarClassifier(FxHaarClassifier** classifier){
	FxHaarClassifier *temp = *classifier;
	FxHaarClassifier *temp_next = NULL;
	while (temp != NULL){
		ReleaseWeakClassifier(temp->classifier);
		temp_next = temp->next;
		free(temp);
		temp = temp_next;
	}
	*classifier = NULL;
}



int fxAdaboostTest(AdaBoostClassifier classifier,FxMat * mat,FxHaarFeature haar,double threshold){
	int feature=fxHaar(mat, haar);
	double weight = 0.0;
	for (int i = 0; i < classifier.ClassifierSizeType0; i++){
		if (feature < classifier.ClassifierType0[i]){
			weight += classifier.ClassifierWeight0[i];
		}
		else{
			weight -= classifier.ClassifierWeight0[i];
		}
	}
	for (int i = 0; i < classifier.ClassifierSizeType1; i++){
		if (feature >= classifier.ClassifierType1[i]){
			weight += classifier.ClassifierWeight1[i];
		}
		else{
			weight -= classifier.ClassifierWeight1[i];
		}
	}
	return weight>threshold ? 1 : 0;
}


void fxQuikSort(FxHaarClassifier ** classifier,int left ,int right){
	if (left < right){
		int i = left, j = right;
		
		FxHaarClassifier *x = classifier[left];
		while (i < j){
			while (i < j && (classifier[j])->True_negitave_rate <= x->True_negitave_rate){
				j--;
			}
			if (i < j){
				classifier[i] = classifier[j];
				i++;
			}
			while (i < j && (classifier[i])->True_negitave_rate > x->True_negitave_rate){
				i++;
			}
			if (i < j){
				classifier[j] = classifier[i];
				j--;
			}
		}
		classifier[i] = x;
		
		fxQuikSort(classifier, left, i-1);
		fxQuikSort(classifier, i + 1, right);
	}
	
}
//typedef struct FxHaarClassifier_{
//	AdaBoostClassifier classifier;
//	FxHaarFeature haar;
//	double Haar_Threshold;
//	double True_negitave_rate;
//	struct FxHaarClassifier_ *next;
//}FxHaarClassifier;
//====================================================================
//struct AdaBoostClassifier_ {
//	int ClassifierSizeType0;
//	int ClassifierSizeType1;
//	int * ClassifierType0;
//	int * ClassifierType1;//if data>=classifier&&data.label==1 type=1;else if data<classifier &&data.label==1 type=0
//	double *ClassifierWeight0;
//	double *ClassifierWeight1;
//};

void fxSaveCasCadeFile(FxHaarClassifier *classifier_list,int stage,char * filename){
	FILE * file = fopen(filename, "w");
	//stage size
	FxHaarClassifier *loop_temp = classifier_list;
	fprintf(file, "%d\n", stage);
	
	while(loop_temp!=NULL){
		fprintf(file, "%d %d %d %d %d\n", 
			loop_temp->haar.offset.x, 
			loop_temp->haar.offset.y, 
			loop_temp->haar.size.width, 
			loop_temp->haar.size.height, 
			loop_temp->haar.type);

		fprintf(file, "%lf %lf\n", loop_temp->Haar_Threshold, loop_temp->True_negitave_rate);

		fprintf(file, "%d %d\n", loop_temp->classifier.ClassifierSizeType0,
			loop_temp ->classifier.ClassifierSizeType1);

		for (int j = 0; j < loop_temp->classifier.ClassifierSizeType0; j++){
			fprintf(file, "%d ", loop_temp->classifier.ClassifierType0[j]);
		}
		fprintf(file, "\n");
		for (int j = 0; j < loop_temp->classifier.ClassifierSizeType0; j++){
			fprintf(file, "%lf ", loop_temp->classifier.ClassifierWeight0[j]);
		}
		fprintf(file, "\n");
		for (int j = 0; j < loop_temp->classifier.ClassifierSizeType1; j++){
			fprintf(file, "%d ", loop_temp->classifier.ClassifierType1[j]);
		}
		fprintf(file, "\n");
		for (int j = 0; j < loop_temp->classifier.ClassifierSizeType1; j++){
			fprintf(file, "%lf ", loop_temp->classifier.ClassifierWeight1[j]);
		}
		fprintf(file, "\n");
		
		loop_temp = loop_temp->next;
	}
	fclose(file);
	
}
int fxClearNegativeSample(FxHaarClassifier classifier,FxMat** mat,int matsize){
	int clear_num = 0;
	for (int i = 0; i < matsize; i++){
		if (mat[i] == NULL)
			continue;
		if (0 == fxAdaboostTest(classifier.classifier, 
			mat[i], 
			classifier.haar,
			classifier.Haar_Threshold)){
			fxReleaseMat(&mat[i]);
			clear_num++;
		}
	}
	return clear_num;
}


void fxCasCade(FxHaarClassifier* classifier, int classifiersize, FxMat *neg_integral_mat,int mat_size,char *path){
	FxHaarClassifier ** classifier_arry = (FxHaarClassifier **)malloc(sizeof(FxHaarClassifier *)*classifiersize);
	FxHaarClassifier * loop_temp = classifier;
	for (int i = 0; i < classifiersize; i++){
		classifier_arry[i] = loop_temp;
		loop_temp = loop_temp->next;
		classifier_arry[i]->next = NULL;
	}
	loop_temp = NULL;
	//sort:
	int neg_mat_num = mat_size;
	fxQuikSort(classifier_arry, 0, classifiersize-1);
	char file_name[100];
	
	for (int i = 0, j = 0; i < FX_CASCADE_MAX_STAGE&&j<classifiersize; i++){
		/***********************************************************************************/
		printf("\n=========================================================\n");
		printf("===============Cascade Stage %d complete=================\n",i);
		printf("=========================================================\n");
		/***********************************************************************************/
		sprintf(file_name, "%s%d%s", path, i, ".cascade");
		int cleared_negnum = 0;
		FxHaarClassifier ** classifier_pointer_loop_temp = NULL;
		//FxHaarClassifier * classifier_pointer_stage = classifier_arry[j];
		FxHaarClassifier * classifier_pointer_stage = NULL;
		while (cleared_negnum < FX_CASCADE_STAGE_THRESHOLD*neg_mat_num&& 
				j<classifiersize&&
				neg_mat_num>0){
			
			int clear_num_classifier= fxClearNegativeSample(*(classifier_arry[j]), neg_integral_mat, mat_size);
			if (clear_num_classifier == 0){
				fxReleaseHaarClassifier(&(classifier_arry[j]));
				j++;
				continue;
			}
			
			if (classifier_pointer_stage == NULL)
				classifier_pointer_stage = classifier_arry[j];
			
			
			if (classifier_pointer_loop_temp != NULL){
				*classifier_pointer_loop_temp = classifier_arry[j];
			}
			cleared_negnum += clear_num_classifier;
			classifier_pointer_loop_temp = &(classifier_arry[j]->next);
			j++;
			
		}
		neg_mat_num -= cleared_negnum;
		if (classifier_pointer_stage != NULL){
			fxSaveCasCadeFile(classifier_pointer_stage, i, file_name);
			fxReleaseHaarClassifier(&classifier_pointer_stage);
		}
		if (neg_mat_num <= 0)
			break;
		
	}
	free(classifier_arry);
	/***********************************************************************************/
	printf("\n=========================================================\n");
	printf("===============Cascade Stage ALL complete================\n");
	printf("=========================================================\n");
	/***********************************************************************************/
}







void fxPostprocessing(AdaBoostTrainData data, FxHaarClassifier * classifier,int neg_num){

	AdaBoostClassifier adaboost = classifier->classifier;
	double threshold = DBL_MAX;
	for (int i = 0; i < data.DataSize - neg_num; i++){
		double i_threshold = 0;
		
		for (int type0_i = 0; type0_i < adaboost.ClassifierSizeType0; type0_i++){
			if (data.data[i] < adaboost.ClassifierType0[type0_i])
				i_threshold += adaboost.ClassifierWeight0[type0_i];
			else 
				i_threshold -= adaboost.ClassifierWeight0[type0_i];
		}
		for (int type1_i = 0; type1_i < adaboost.ClassifierSizeType1; type1_i++){
			if (data.data[i] >= adaboost.ClassifierType1[type1_i])
				i_threshold += adaboost.ClassifierWeight1[type1_i];
			else
				i_threshold -= adaboost.ClassifierWeight1[type1_i];
		}

		threshold = (i_threshold < threshold) ? i_threshold : threshold;
		
	}
	classifier->Haar_Threshold =threshold;

	int true_negative = 0;
	for (int i = data.DataSize-neg_num; i < data.DataSize; i++){
		double i_threshold = 0;
		for (int type0_i = 0; type0_i < adaboost.ClassifierSizeType0; type0_i++){
			if (data.data[i] < adaboost.ClassifierType0[type0_i])
				i_threshold += adaboost.ClassifierWeight0[type0_i];
			else
				i_threshold -= adaboost.ClassifierWeight0[type0_i];
		}
		for (int type1_i = 0; type1_i < adaboost.ClassifierSizeType1; type1_i++){
			if (data.data[i] >= adaboost.ClassifierType1[type1_i])
				i_threshold += adaboost.ClassifierWeight1[type1_i];
			else
				i_threshold -= adaboost.ClassifierWeight1[type1_i];

		}

		if (i_threshold < threshold)
			true_negative++;
	}
	classifier->True_negitave_rate = (double)true_negative / neg_num;

}

void fxHaarTrain(char *sample_path, char *cascadfilepath, int pos_num, int neg_num,FxSize size){
	
	//Step 1
	//
	//
	char pos_path[100];
	char neg_path[100];
	char pos_name[100];
	char neg_name[100];
	int classifiersize = 0;
	sprintf(pos_path, "%s%s", sample_path, "posdata\\");
	sprintf(neg_path, "%s%s", sample_path, "negdata\\image_");
	FxMat ** pos_integral_mat = (FxMat**)malloc(sizeof(FxMat*)*pos_num);
	FxMat ** neg_integral_mat = (FxMat**)malloc(sizeof(FxMat*)*neg_num);
	for (int pos_i = 0; pos_i < pos_num; pos_i++){
		sprintf(pos_name, "%s%d%s", pos_path, pos_i, ".jpg");
		FxMat* src_mat= readImage(pos_name, 0);
		pos_integral_mat[pos_i] = fxCreateMat(fxSize(src_mat->width,src_mat->height),FX_32C1);
		fxIntegralImage(src_mat, pos_integral_mat[pos_i]);
		fxReleaseMat(&src_mat);
		printf("pos_mat integral finish:%lf%%\r", (double)pos_i / (pos_num-1) * 100.0);
		
	}
	printf("\n");
	for (int neg_i = 0; neg_i < neg_num; neg_i++){
		sprintf(neg_name, "%s%d%s", neg_path, neg_i, ".jpg");
		FxMat* src_mat = readImage(neg_name, 0);
		neg_integral_mat[neg_i] = fxCreateMat(fxSize(src_mat->width, src_mat->height), FX_32C1);
		fxIntegralImage(src_mat, neg_integral_mat[neg_i]);
		fxReleaseMat(&src_mat);
		printf("neg_mat integral finish:%lf%%\r", (double)neg_i / (neg_num-1) * 100.0);
	}
	printf("\n");

	//step2
	//
	//
	FxHaarClassifier* classifier = (FxHaarClassifier*)malloc(sizeof(FxHaarClassifier));
	classifier->next = NULL;
	FxHaarClassifier **temp = &classifier;
	//
	int sample_width = size.width;
	int sample_height = size.height;
	//step3
	//
	//
	AdaBoostTrainData haar_data;
	haar_data.DataSize = pos_num + neg_num;
	haar_data.data = (int *)malloc(sizeof(int)*(pos_num + neg_num));
	haar_data.label = (char *)malloc(sizeof(char)*(pos_num + neg_num));
	printf("adaboosting");
/*****************************************************************************************/
	

	for (int type = FX_HAAR_TYPE1; type <= FX_HAAR_TYPE4; type++){
		for (int h = 3; h < sample_height-3; h++){
			if ((type == FX_HAAR_TYPE1&&h % 2 != 0) ||
				(type == FX_HAAR_TYPE3&&h % 2 != 0))
				continue;
			for (int w = 3; w < sample_width-3; w++){
				if ((type == FX_HAAR_TYPE2&&w % 2 != 0) || 
					(type == FX_HAAR_TYPE3&&w % 2 != 0) ||
					(type == FX_HAAR_TYPE4&&w % 3 != 0) 
					)
					continue;
				for (int x = 0; x < sample_width-w; x++){
					
					for (int y = 0; y < sample_height-h; y++){
						
						FxHaarFeature haar_feature;
						haar_feature.offset.x = x;
						haar_feature.offset.y = y;
						haar_feature.size.width = w;
						haar_feature.size.height = h;
						haar_feature.type = type;
						int total_num = 0;
						for (int pos_i = 0; pos_i < pos_num; pos_i++, total_num++){
							haar_data.data[total_num] = fxHaar(pos_integral_mat[pos_i], haar_feature);
							haar_data.label[total_num] = 1;
							//printf("pos_data:%d\n", haar_data.data[total_num]);
						}
						for (int neg_i = 0; neg_i < neg_num; neg_i++, total_num++){
							haar_data.data[total_num] = fxHaar(neg_integral_mat[neg_i], haar_feature);
							haar_data.label[total_num] = 0;
							//printf("neg_data:%d\n", haar_data.data[total_num]);
						}
						
						//
						//
						//
						if ((*temp) == NULL){
							(*temp) = (FxHaarClassifier*)malloc(sizeof(FxHaarClassifier));
							(*temp)->next = NULL; 
						}
						(*temp)->classifier=AdaBoost(haar_data, FX_ADABOOST_CLASSIFIER_SIZE);
						(*temp)->haar = haar_feature;
						
						printf(".");
						//
						//
						//
						fxPostprocessing(haar_data, (*temp),neg_num);
						
						if ((*temp)->True_negitave_rate > FX_MIN_TRUE_NEGATIVE_RATE){
							
							printf("\nHaar type:%d x:%d y:%d w:%d h:%d\n", haar_feature.type, haar_feature.offset.x, haar_feature.offset.y, haar_feature.size.width, haar_feature.size.height);
							printf("Classifier : threshold %lf true negative rate :%lf\n", (*temp)->Haar_Threshold, (*temp)->True_negitave_rate);
							classifiersize++;
							temp = &((*temp)->next);
							printf("\nadaboosting");
							
						}
					}
				
				}
			}
		}
		
	}
/*****************************************************************************************/
	//
	//
	//cascade
	for (int i = 0; i < pos_num; i++){
		fxReleaseMat(&pos_integral_mat[i]);
	}
	fxCasCade(classifier, classifiersize, neg_integral_mat,neg_num ,cascadfilepath);
	for (int i = 0; i < neg_num; i++){
		if (neg_integral_mat[i]!=NULL)
			fxReleaseMat(&neg_integral_mat[i]);
	}
	//fxReleaseHaarClassifier(&classifier);
}