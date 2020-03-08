//shape index different
#include "faSID.h"
static int faRandomRadius(int stage_num){
	int radius = (STAGE - stage_num)+10;
	return radius;
}

static void faRandomOffset(FxPoint* off_set_array, int stage_num){
	int radius = faRandomRadius(stage_num);
	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(OPENMP_THREAD_NUM)
#endif
	for (loop_i = 0; loop_i < SID_RANDOM_NUM*2; loop_i++){
		off_set_array[loop_i].x = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		off_set_array[loop_i].y = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		
	}
	
}

int faSIDFeature(FxMat *mat , FxPoint64 landmark , FxPoint offset1 , FxPoint offset2){
	int width = mat->width;
	int height = mat->height;
	FxPoint point1, point2;
	point1.x = (int)landmark.x + offset1.x;
	point1.y = (int)landmark.y + offset1.y;
	point2.x = (int)landmark.x + offset2.x;
	point2.y = (int)landmark.y + offset2.y;
	double a, b;
	//out of edge set to edge
	if ( point1.x <= 0 ){
		point1.x = 0;
	}
	else if (point1.x >= width){
		point1.x = width-1;
	}
	if (point1.y >= height){
		point1.y = height-1;
	}
	else if (point1.y < 0){
		point1.y = 0;
	}
	a = fxGetRealData(mat, point1.x, point1.y);


	if (point2.x >= width)
		point2.x = width - 1;
	else if (point2.x <= 0)
		point2.x = 0;

	if (point2.y >= height)
		point2.y = height - 1;
	else if(point2.y < 0)
		point2.y = 0;
	

	b = fxGetRealData(mat, point2.x, point2.y);
	
	return (int)(a - b);
}






double faVariance(FaDataInNode* data_inNode, FxPoint *off_set_array,int landmark_num,int * threshold){
	if (data_inNode->fadata_size==0){
		*threshold = MIN_SID_FEATURE;
		return DBL_MAX;
	}
	int* sid_feature = (int *)fxMalloc(sizeof(int)*data_inNode->fadata_size);
	//
	for (int i = 0; i < data_inNode->fadata_size; i++){
		FxPoint64 landmark;
		landmark.x = data_inNode->fadata_p_array[i]->landmark_realtime[landmark_num].x;
		landmark.y = data_inNode->fadata_p_array[i]->landmark_realtime[landmark_num].y;
		sid_feature[i] = faSIDFeature(data_inNode->fadata_p_array[i]->image,
			landmark, off_set_array[0], off_set_array[1]);

	}
	int random_threshold_num = fxRandom(data_inNode->fadata_size, FX_RANDOM_MEAN_HALF_N);
	*threshold = sid_feature[random_threshold_num];


	double min_variance = DBL_MAX;
	
	
	double left_dx = 0, left_dy = 0, right_dx = 0, right_dy = 0;
	double right_mx = 0, right_my = 0, left_mx = 0, left_my = 0;
	int left_num = 0;
	int right_num = 0;
	for (int i = 0; i < data_inNode->fadata_size; i++){
		if (sid_feature[i] < *threshold){
			left_mx += data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].x;
			left_my += data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].y;
			left_num++;
		}
		else{
			right_mx += data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].x;
			right_my += data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].y;
			right_num++;
		}

	}


	if (left_num != 0){
		left_mx /= left_num;
		left_my /= left_num;
	}
	if (right_num != 0){
		right_mx /= right_num;
		right_my /= right_num;
	}


	for (int i = 0; i < data_inNode->fadata_size; i++){
		if (sid_feature[i] < *threshold){
			left_dx += (data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].x - left_mx)*
				(data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].x - left_mx);
			left_dy += (data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].y - left_my)*
				(data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].y - left_my);
		}
		else{
			right_dx += (data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].x - right_mx)*
				(data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].x - right_mx);
			right_dy += (data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].y - right_my)*
				(data_inNode->fadata_p_array[i]->landmark_delta[landmark_num].y - right_my);

		}
	}
	
	double left_variance = (left_dx + left_dy) ;/// (left_num == 0 ? 1 : left_num);
	double right_variance = (right_dx + right_dy);// / (right_num == 0 ? 1 : right_num);
	double variance = (left_variance + right_variance);
		


	
	fxFree(&sid_feature);
	return variance;

}


FaSIDFeature faSID(FaTrainData *traindata, int tree_position,int landmark_num, int stage_num){
	FaSIDFeature sid_feature;
	FxPoint off_set_array[SID_RANDOM_NUM*2];
	FaDataInNode* data_inNode = faDataInNode(traindata, tree_position);


	faRandomOffset(off_set_array, stage_num);


	double variance[SID_RANDOM_NUM];
	int min_variance_threshold[SID_RANDOM_NUM];
	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for num_threads(OPENMP_THREAD_NUM)
#endif
	for (loop_i = 0; loop_i < SID_RANDOM_NUM; loop_i++){
		int min_variance_threshold_temp=0.0;
		variance[loop_i] = faVariance(data_inNode, 
			(FxPoint*)&(off_set_array[loop_i * 2]), 
			landmark_num, 
			&(min_variance_threshold_temp));
		min_variance_threshold[loop_i] = min_variance_threshold_temp;
		
	}

	double min_variance_v = DBL_MAX;
	int min_variance_num = 0;
	for (int i = 0; i < SID_RANDOM_NUM; i++){
		if (variance[i] < min_variance_v){
			min_variance_v = variance[i];
			min_variance_num = i;
		}
		//printf("(%d,%d) (%d,%d) %lf \n", off_set_array[i * 2].x, off_set_array[i * 2].y, 
		//	off_set_array[i * 2 + 1].x, off_set_array[i * 2 + 1].y, variance[i]);
	}


	sid_feature.off_set[0].x = off_set_array[min_variance_num * 2].x;
	sid_feature.off_set[0].y = off_set_array[min_variance_num * 2].y;
	sid_feature.off_set[1].x = off_set_array[min_variance_num * 2 + 1].x;
	sid_feature.off_set[1].y = off_set_array[min_variance_num * 2 + 1].y;
	sid_feature.threshold = min_variance_threshold[min_variance_num]; 




	int go2left = 0;
	int go2right = 0;



	for (int i = 0; i < data_inNode->fadata_size; i++){

		FaData* sample=data_inNode->fadata_p_array[i];
		int feature = faSIDFeature(sample->image, sample->landmark_realtime[landmark_num],
			sid_feature.off_set[0],sid_feature.off_set[1]);

		if (feature < sid_feature.threshold){
			sample->tree_position = LEFT_CHILD  (tree_position);
			go2left++;
		}
		else{
			sample->tree_position = RIGHT_CHILD (tree_position);
			go2right++;
		}
	
	}
	//printf("left samples num:%d           right samples num:%d \n", go2left, go2right);
	faFreeDataInNode(&data_inNode);
	return sid_feature;
}





FaDataInNode * faDataInNode(FaTrainData *traindata, int position) {
	FaDataInNode* data_innode = (FaDataInNode*)fxMalloc(sizeof(FaDataInNode));
	FaData** data_p_array = (FaData**)fxMalloc(sizeof(FaData*)*traindata->data_size);
	int k = 0;
	for (int i = 0; i < traindata->data_size; i++){
		if (traindata->data_array[i].tree_position == position){
			data_p_array[k] = &(traindata->data_array[i]);
			k++;
		}
	}


	data_innode->fadata_size = k;
	data_innode->fadata_p_array = (FaData **)fxMalloc(sizeof(FaData*)*k);
	memcpy(data_innode->fadata_p_array,data_p_array,sizeof(FaData*)*k);
	fxFree(&data_p_array);
	return data_innode;
}



void faFreeDataInNode(FaDataInNode ** fadata_innode){
	fxFree(&((*fadata_innode)->fadata_p_array));
	fxFree(fadata_innode);
}