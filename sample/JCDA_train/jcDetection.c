#include "jcForest.h"


int jcDetectionFeatureValue(FxMat * mat, FxPoint * center, FxPoint * off_set){ 
	int a = MAX_FEATURE_VALUE;
	int b = MAX_FEATURE_VALUE;
	int width = mat->width;
	int height = mat->height;
	int off_set0_x = center[0].x + off_set[0].x;
	int off_set0_y = center[0].y + off_set[0].y;
	int off_set1_x = center[1].x + off_set[1].x;
	int off_set1_y = center[1].y + off_set[1].y;
	//printf("off_set0 :(%d,%d)  off_set1 :(%d,%d)\n", off_set0_x, off_set0_y, off_set1_x, off_set1_y);
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


double jcEntropy(int pos_left,int pos_right,int neg_left,int neg_right){
	double log_2 = log(2.0);
	int left_num = pos_left + neg_left;
	int right_num = pos_right + neg_right;
	double p_pos_right=0.0;
	double p_left=0.0;
	double p_pos_left=0.0;
	if (left_num+right_num != 0)
		p_left = (double)left_num / (left_num + right_num);
	if (left_num!=0)
		p_pos_left = (double)(pos_left) / left_num;
	if (right_num!=0)
		p_pos_right = (double)(pos_right) / right_num;
	//entropy=real_entropy/log(2.0)
	double h0 ;
	double h1 ;

	double h2 ;
	double h3 ;
	if (p_pos_left != 0)
		h0 = p_pos_left*log(p_pos_left) ;
	else
		h0 = 0.0;
	if (p_pos_left != 1.0)
		h1 = (1.0 - p_pos_left)*log((1.0 - p_pos_left)) ;
	else
		h1 = 0.0;
	if (p_pos_right != 0)
		h2 = p_pos_right*log(p_pos_right) ;
	else
		h2 = 0.0;
	if (p_pos_right != 1.0)
		h3 = (1.0 - p_pos_right)*log(1.0 - p_pos_right );
	else
		h3 = 0.0;

	return (p_left*(h0 + h1) + (1.0 - p_left)*(h2 + h3) )/ log_2*(-1.0);

}
void jcDetectionThreshold(JcTrainData*	train_data, JcSampleSet *sample_set, JcDetectionFeature *dt_feature){
	int pos_num = sample_set->pos_size;
	int neg_num = sample_set->neg_size;
	int facial_point0 = dt_feature->FacialPoint[0];
	int facial_point1 = dt_feature->FacialPoint[1];


	int * pos_feature_value = (int *)fxMalloc(sizeof(int)*pos_num);
	int * neg_feature_value = (int *)fxMalloc(sizeof(int)*neg_num);
	//double * entropy = (double *)fxMalloc(sizeof(double)*(MAX_FEATURE_VALUE)* 2);

	
	int random_threshold_num = fxRandom(pos_num+neg_num, FX_RANDOM_MEAN_HALF_N);
	int threshold = 0;
	//random_threshold_num %2 ==1 pos else neg to get a threshold;
	//int threshold = (random_threshold_num <pos_num) ? 
	//	(pos_feature_value[(random_threshold_num)]) : (neg_feature_value[(random_threshold_num - pos_num)]);
	if (random_threshold_num <pos_num){
		int sample_num = sample_set->pos_positions[random_threshold_num];
		JcTrainSample *train_sample = &(train_data->positive_sample[sample_num]);
		FxPoint center[2];

		center[0].x = (int)((train_sample->Face_Shap)->LandMark)[facial_point0].x;
		center[0].y = (int)((train_sample->Face_Shap)->LandMark)[facial_point0].y;
		center[1].x = (int)((train_sample->Face_Shap)->LandMark)[facial_point1].x;
		center[1].y = (int)((train_sample->Face_Shap)->LandMark)[facial_point1].y;

		threshold = jcDetectionFeatureValue(train_sample->mat, center, dt_feature->off_set);

	}
	else{
		int sample_num = sample_set->neg_positions[random_threshold_num - pos_num];
		JcTrainSample *train_sample = &(train_data->negative_sample[sample_num]);
		FxPoint center[2];

		center[0].x = (int)((train_sample->Face_Shap)->LandMark)[facial_point0].x;
		center[0].y = (int)((train_sample->Face_Shap)->LandMark)[facial_point0].y;
		center[1].x = (int)((train_sample->Face_Shap)->LandMark)[facial_point1].x;
		center[1].y = (int)((train_sample->Face_Shap)->LandMark)[facial_point1].y;

		threshold = jcDetectionFeatureValue(train_sample->mat, center, dt_feature->off_set);

	}



	
	// less than loop_i is left
	// else is right
	int pos_left = 0;
	int neg_left = 0;
	int pos_right = 0;
	int neg_right = 0;
	for (int i = 0; i < pos_num; i++){
		if (pos_feature_value[i] < threshold)
			pos_left++;
		else
			pos_right++;
	}
	for (int i = 0; i < neg_num; i++){
		if (neg_feature_value[i] < threshold)
			neg_left++;
		else
			neg_right++;
		
	}
	double entropy = jcEntropy(pos_left, pos_right, neg_left, neg_right);
	
	
	dt_feature->min_entropy = entropy;
	dt_feature->threshold = threshold;
	//printf("\n\n%%%%%%%%%% detection threshold free %%%%%%%%%%%\n\n");
	fxFree(&pos_feature_value);
	fxFree(&neg_feature_value);

}
void jcSplitDetectionNode(JcTrainData* train_data, JcSampleSet *sample_set, int position, JcDetectionFeature *dt_feature){
	int pos_num = sample_set->pos_size;
	int neg_num = sample_set->neg_size;
	int dt_FeatureValue;
	int facial_point0 = dt_feature->FacialPoint[0];
	int facial_point1 = dt_feature->FacialPoint[1];

	for (int i = 0; i < pos_num; i++){
		int sample_num = sample_set->pos_positions[i];
		JcTrainSample *train_sample = &(train_data->positive_sample[sample_num]);
		FxPoint center[2];

		center[0].x = (int) ((train_sample->Face_Shap)->LandMark[facial_point0].x);
		center[0].y = (int) ((train_sample->Face_Shap)->LandMark[facial_point0].y);
		center[1].x = (int) ((train_sample->Face_Shap)->LandMark[facial_point1].x);
		center[1].y = (int) ((train_sample->Face_Shap)->LandMark[facial_point1].y);
		dt_FeatureValue = jcDetectionFeatureValue(train_sample->mat, center, dt_feature->off_set);
		//////////////////////////////////////////////////////////////////////////////////////////
		//int off_set0_x = center[0].x + dt_feature->off_set[0].x;
		//int off_set0_y = center[0].y + dt_feature->off_set[0].y;
		//int off_set1_x = center[1].x + dt_feature->off_set[1].x;
		//int off_set1_y = center[1].y + dt_feature->off_set[1].y;
		//printf("off_set0 :(%4d,%4d)  off_set1 :(%4d,%4d)   ----->value: %d\n",
		//	off_set0_x, off_set0_y, off_set1_x, off_set1_y, dt_FeatureValue);
		//////////////////////////////////////////////////////////////////////////////////////////
		if (dt_FeatureValue < dt_feature->threshold)
			train_sample->Position_in_Tree = jcTreeLeftChild(position);
		else
			train_sample->Position_in_Tree = jcTreeRightChild(position);
	}

	//printf("======================================test==============================================");
	//
	//
	//
	for (int i = 0; i < neg_num; i++){
		int sample_num = sample_set->neg_positions[i];
		JcTrainSample *train_sample = &(train_data->negative_sample[sample_num]);
		FxPoint center[2];

		center[0].x = (int) ((train_sample->Face_Shap)->LandMark)[facial_point0].x;
		center[0].y = (int) ((train_sample->Face_Shap)->LandMark)[facial_point0].y;
		center[1].x = (int) ((train_sample->Face_Shap)->LandMark)[facial_point1].x;
		center[1].y = (int) ((train_sample->Face_Shap)->LandMark)[facial_point1].y;
		dt_FeatureValue = jcDetectionFeatureValue(train_sample->mat, center, dt_feature->off_set);
		//////////////////////////////////////////////////////////////////////////////////////////
		//int off_set0_x = center[0].x + dt_feature->off_set[0].x;
		//int off_set0_y = center[0].y + dt_feature->off_set[0].y;
		//int off_set1_x = center[1].x + dt_feature->off_set[1].x;
		//int off_set1_y = center[1].y + dt_feature->off_set[1].y;
		//printf("off_set0 :(%4d,%4d)  off_set1 :(%4d,%4d)   ----->value: %d\n",
		//	off_set0_x, off_set0_y, off_set1_x, off_set1_y, dt_FeatureValue);
		//////////////////////////////////////////////////////////////////////////////////////////
		//printf("%d ", dt_FeatureValue);
		if (dt_FeatureValue < dt_feature->threshold)
			train_sample->Position_in_Tree = jcTreeLeftChild(position);
		else
			train_sample->Position_in_Tree = jcTreeRightChild(position);
	}
	//printf("======================================test==============================================");
	//


}

void jcLearnDetectionNode(JcTrainData* train_data, int position, int cascade_stage, JcDetectionFeature *feature){
	JcSampleSet *sampleset = jcSampleSet(train_data, position);
	if (sampleset->neg_size != 0 && sampleset->pos_size != 0){

		uint32 radiu = jcA_DFeatureRadius(cascade_stage);
		JcDetectionFeature *df_array = jcCreateDF_Array(INTERAL_TEST);
		jcRandDeFeature(df_array, INTERAL_TEST, radiu);
		int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif
		for (loop_i = 0; loop_i < INTERAL_TEST; loop_i++){
			jcDetectionThreshold(train_data, sampleset, &(df_array[loop_i]));
		}

		double min_entropy = DBL_MAX;
		int min_entropy_position = -1;
		for (int i = 0; i < INTERAL_TEST; i++){
			//printf("%lf \n", df_array[i].min_entropy);
			if (df_array[i].min_entropy < min_entropy){
				min_entropy = df_array[i].min_entropy;
				min_entropy_position = i;
			}
		}

		feature->FacialPoint[0] = df_array[min_entropy_position].FacialPoint[0];
		feature->FacialPoint[1] = df_array[min_entropy_position].FacialPoint[1];
		feature->min_entropy = min_entropy;
		feature->off_set[0].x = df_array[min_entropy_position].off_set[0].x;
		feature->off_set[0].y = df_array[min_entropy_position].off_set[0].y;
		feature->off_set[1].x = df_array[min_entropy_position].off_set[1].x;
		feature->off_set[1].y = df_array[min_entropy_position].off_set[1].y;
		feature->threshold = df_array[min_entropy_position].threshold;
		jcSplitDetectionNode(train_data, sampleset, position, feature);
#ifdef OURPUTPRINT	
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		printf("learn a detection node ,tree position %d \n", position);
		printf("feature facial point <%d , %d> :(%d ,%d ),(%d ,%d)\n", feature->FacialPoint[0], feature->FacialPoint[1],
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		printf("pos_size :%d    neg_size:%d\n", sampleset->pos_size, sampleset->neg_size);
		printf("min entropy :%lf\n", feature->min_entropy);
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
#ifdef DEBUG	
		fprintf(logfile,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		fprintf(logfile, "learn a detection node ,tree position %d \n", position);
		fprintf(logfile, "feature facial point <%d , %d> :(%d ,%d ),(%d ,%d)\n", feature->FacialPoint[0], feature->FacialPoint[1],
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		fprintf(logfile, "pos_size :%d    neg_size:%d\n", sampleset->pos_size, sampleset->neg_size);
		fprintf(logfile, "min entropy :%lf\n", feature->min_entropy);
		fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
		jcFreeDF_Array(&df_array);
		jcFreeSampleSet(&sampleset);
	}
	else{
#ifdef OURPUTPRINT
		if (sampleset->neg_size == 0)
			printf("sampleset nega size 0 \n");
		if (sampleset->pos_size == 0)
			printf("sampleset posi size 0 \n");
#endif
		feature->FacialPoint[0] = 0;
		feature->FacialPoint[1] = 0;
		feature->min_entropy = 0;
		feature->off_set[0].x = 0;
		feature->off_set[0].y = 0;
		feature->off_set[1].x = 0;
		feature->off_set[1].y =0;
		feature->threshold =MAX_FEATURE_VALUE;
		jcSplitDetectionNode(train_data, sampleset, position, feature);
#ifdef OURPUTPRINT

		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		printf("learn a detection node ,tree position %d \n", position);
		printf("feature facial point <%d , %d> :(%d ,%d ),(%d ,%d)\n", feature->FacialPoint[0], feature->FacialPoint[1],
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		printf("pos_size :%d    neg_size:%d\n", sampleset->pos_size, sampleset->neg_size);
		printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif	
#ifdef DEBUG

		fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		fprintf(logfile,  "learn a detection node ,tree position %d \n", position);
		fprintf(logfile,  "feature facial point <%d , %d> :(%d ,%d ),(%d ,%d)\n", feature->FacialPoint[0], feature->FacialPoint[1],
			feature->off_set[0].x, feature->off_set[0].y, feature->off_set[1].x, feature->off_set[1].y);
		fprintf(logfile, "pos_size :%d    neg_size:%d\n", sampleset->pos_size, sampleset->neg_size);
		fprintf(logfile,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif	
	}

	
}