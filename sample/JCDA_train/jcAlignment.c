#include "jcForest.h"
#include <stdio.h>
int jcAlignmentFeatureValue(FxMat * mat, FxPoint64* center, FxPoint * off_set){
	int a = MAX_FEATURE_VALUE;
	int b = MAX_FEATURE_VALUE;
	int width = mat->width;
	int height = mat->height;
	int off_set0_x = (int)center->x + off_set[0].x;
	int off_set0_y = (int)center->y + off_set[0].y;
	int off_set1_x = (int)center->x + off_set[1].x;
	int off_set1_y = (int)center->y + off_set[1].y;
	if (off_set0_x < 0)
		off_set0_x = 0;
	if (off_set0_x >= width)
		off_set0_x = width - 1;
	if (off_set0_y < 0)
		off_set0_y = 0;
	if (off_set0_y >= height)
		off_set0_y = height - 1;
	if (off_set1_x < 0)
		off_set1_x = 0;
	if (off_set1_x >= width)
		off_set1_x = width - 1;
	if (off_set1_y < 0)
		off_set1_y = 0;
	if (off_set1_y >= height)
		off_set1_y = height - 1;

	a = (int)fxGetRealData(mat, off_set0_x, off_set0_y);

	b = (int)fxGetRealData(mat, off_set1_x, off_set1_y);
	return a - b;
}



void jcAlignmentThreshold(JcTrainData*train_data, JcSampleSet *sample_set, int facial_point, JcAlignmentFeature *al_feature){
	int pos_num = sample_set->pos_size;
	if (pos_num == 0)
	{
		printf("wrong!\n");
	}
	

	int * al_Feature_array = (int *)fxMalloc(sizeof(int)*(pos_num));
	for (int i = 0; i < pos_num; i++){
		int sample_num = sample_set->pos_positions[i];
		FxPoint64 center;
		JcTrainSample * train_sample = &((train_data->positive_sample)[sample_num]);
		center.x = (train_sample->Face_Shap)->LandMark[facial_point].x;
		center.y = (train_sample->Face_Shap)->LandMark[facial_point].y;
		al_Feature_array[i] = jcAlignmentFeatureValue(train_sample->mat, &center, al_feature->off_set);
	}
	int rand_threshold = fxRandom(pos_num, FX_RANDOM_MEAN_HALF_N);
	int threshold = al_Feature_array[rand_threshold];
	

	// get mean 
	FxPoint64 delta_mean_left;
	delta_mean_left.x = 0.0;
	delta_mean_left.y = 0.0;


	FxPoint64 delta_mean_right;
	delta_mean_right.x = 0.0;
	delta_mean_right.y = 0.0;

	int left_num = 0;
	int right_num = 0;


	for (int i = 0; i < pos_num; i++){
		int sample_num = sample_set->pos_positions[i];
		JcTrainSample * train_sample = &((train_data->positive_sample)[sample_num]);
		if (al_Feature_array[i] < threshold){
			delta_mean_left.x += train_sample->Face_Shap_Delta->LandMark[facial_point].x;
			delta_mean_left.y += train_sample->Face_Shap_Delta->LandMark[facial_point].y;
			left_num++;
		}
		else{
			delta_mean_right.x += train_sample->Face_Shap_Delta->LandMark[facial_point].x;
			delta_mean_right.y += train_sample->Face_Shap_Delta->LandMark[facial_point].y;
			right_num++;
		}

	}
	if (left_num != 0){
		delta_mean_left.x /= left_num;
		delta_mean_left.y /= left_num;
	}
	if (right_num != 0){
		delta_mean_right.x /= right_num;
		delta_mean_right.y /= right_num;
	}
	//
	//
	//get variance
	double variance = 0.0;
	for (int i = 0; i < pos_num; i++){
		int sample_num = sample_set->pos_positions[i];
		JcTrainSample * train_sample = &((train_data->positive_sample)[sample_num]);
		if (al_Feature_array[i] < threshold){
			double diff_x = (delta_mean_left.x - train_sample->Face_Shap_Delta->LandMark[facial_point].x)*
				(delta_mean_left.x - train_sample->Face_Shap_Delta->LandMark[facial_point].x);
			double diff_y = (delta_mean_left.y - train_sample->Face_Shap_Delta->LandMark[facial_point].y)*
				(delta_mean_left.y - train_sample->Face_Shap_Delta->LandMark[facial_point].y);
			variance += (diff_x + diff_y);
		}
		else{
			double diff_x = (delta_mean_right.x - train_sample->Face_Shap_Delta->LandMark[facial_point].x)*
				(delta_mean_right.x - train_sample->Face_Shap_Delta->LandMark[facial_point].x);
			double diff_y = (delta_mean_right.y - train_sample->Face_Shap_Delta->LandMark[facial_point].y)*
				(delta_mean_right.y - train_sample->Face_Shap_Delta->LandMark[facial_point].y);
			variance += (diff_x + diff_y);
		}

	}
	

	al_feature->threshold = threshold;
	al_feature->min_variance = variance;
	//fxFree(&variance);
	fxFree(&al_Feature_array);
}

void JcSplitAlignmentNode(JcTrainData*train_data, JcSampleSet *sample_set, int facial_point, int position, JcAlignmentFeature *al_feature){
	int pos_num = sample_set->pos_size;
	int neg_num = sample_set->neg_size;

	int al_Featurevalue;

	//split positive samples
	for (int i = 0; i < pos_num; i++){
		int sample_num = sample_set->pos_positions[i];
		FxPoint64 center;
		JcTrainSample * train_sample = &((train_data->positive_sample)[sample_num]);
		center.x = (train_sample->Face_Shap)->LandMark[facial_point].x;
		center.y = (train_sample->Face_Shap)->LandMark[facial_point].y;
		al_Featurevalue = jcAlignmentFeatureValue(train_sample->mat, &center, al_feature->off_set);
		if (al_Featurevalue < al_feature->threshold)
			train_sample->Position_in_Tree = jcTreeLeftChild(position);
		else{
			train_sample->Position_in_Tree = jcTreeRightChild(position);

		}
	}
	//split negative samples
	for (int i = 0; i < neg_num; i++){
		int sample_num = sample_set->neg_positions[i];
		FxPoint64 center;
		JcTrainSample * train_sample = &((train_data->negative_sample)[sample_num]);
		center.x = (train_sample->Face_Shap)->LandMark[facial_point].x;
		center.y = (train_sample->Face_Shap)->LandMark[facial_point].y;
		al_Featurevalue = jcAlignmentFeatureValue(train_sample->mat, &center, al_feature->off_set);
		if (al_Featurevalue < al_feature->threshold)
			train_sample->Position_in_Tree = jcTreeLeftChild(position);
		else{
			train_sample->Position_in_Tree = jcTreeRightChild(position);

		}
	}

}





void jcLearnAlignmentNode(JcTrainData* train_data, int cascade_stage, int facial_point, int position, JcAlignmentFeature *feature){
	JcSampleSet* sampleset = jcSampleSet(train_data, position);
	if (sampleset->pos_size != 0){
		uint32 radiu = jcA_DFeatureRadius(cascade_stage);
		JcAlignmentFeature * al_array = jcCreateAF_Array(INTERAL_TEST);
		jcRandAlFeature(al_array, INTERAL_TEST, radiu);
		int loop_i = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (loop_i = 0; loop_i < INTERAL_TEST; loop_i++){
			jcAlignmentThreshold(train_data, sampleset, facial_point, &(al_array[loop_i]));

		}
		int min_variance_feature_num = 0;
		double min_variance_value = DBL_MAX;
		for (int i = 0; i < INTERAL_TEST; i++){
			if (min_variance_value >= al_array[i].min_variance){
				min_variance_value = al_array[i].min_variance;
				min_variance_feature_num = i;
			}
		}
		feature->min_variance = al_array[min_variance_feature_num].min_variance;
		feature->off_set[0].x = al_array[min_variance_feature_num].off_set[0].x;
		feature->off_set[0].y = al_array[min_variance_feature_num].off_set[0].y;
		feature->off_set[1].x = al_array[min_variance_feature_num].off_set[1].x;
		feature->off_set[1].y = al_array[min_variance_feature_num].off_set[1].y;
		feature->threshold = al_array[min_variance_feature_num].threshold;
		JcSplitAlignmentNode(train_data, sampleset, facial_point, position, feature);

#ifdef OURPUTPRINT
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		printf("learn a alignment node ,tree position %d \n", position);
		printf("feature facial point <%d> :(%d ,%d ),(%d ,%d)\n", facial_point,
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		printf("pos_size :%d    neg_size:%d\n", sampleset->pos_size, sampleset->neg_size);
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
#ifdef DEBUG
		fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		fprintf(logfile, "learn a alignment node ,tree position %d \n", position);
		fprintf(logfile, "feature facial point <%d> :(%d ,%d ),(%d ,%d)\n", facial_point,
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		fprintf(logfile, "pos_size :%d    neg_size:%d\n", sampleset->pos_size, sampleset->neg_size);
		fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
		jcFreeAF_Array(&al_array);
		jcFreeSampleSet(&sampleset);
	}
	else{
		feature->min_variance = -1.0;
		feature->off_set[0].x = 0;
		feature->off_set[0].y = 0;
		feature->off_set[1].x =	0;
		feature->off_set[1].y = 0;
		feature->threshold = MAX_FEATURE_VALUE;
#ifdef OURPUTPRINT
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		printf("learn a alignment node ,tree position %d \n", position);
		printf("feature facial point <%d> :(%d ,%d ),(%d ,%d)\n", facial_point,
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
#ifdef DEBUG
		fprintf(logfile,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		fprintf(logfile, "learn a alignment node ,tree position %d \n", position);
		fprintf(logfile, "feature facial point <%d> :(%d ,%d ),(%d ,%d)\n", facial_point,
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
		JcSplitAlignmentNode(train_data, sampleset, facial_point, position, feature);
	}


}