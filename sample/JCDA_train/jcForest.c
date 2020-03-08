#include "jcForest.h"




void jcInitDataPose(JcTrainData *train_data){
	int neg_num = train_data->negative_num;
	int pos_num = train_data->positive_num;

	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < neg_num; loop_i++){
		train_data->negative_sample[loop_i].Position_in_Tree = 0;
	}


#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < pos_num; loop_i++){
		train_data->positive_sample[loop_i].Position_in_Tree = 0;
	}

}


JcTreeNode *jcCreateTreeNode(JcTreeNodeType node_type){
	JcTreeNode *node = (JcTreeNode *)fxMalloc(sizeof(JcTreeNode));
	node->node_type = node_type;
	switch (node_type){
		case ALIGNMENT:
		{
			node->feature_al = (JcAlignmentFeature*)fxMalloc(sizeof(JcAlignmentFeature));
			node->feature_dt = NULL;
			node->feature_lf = NULL;
			break;
		}
		case DETECTION:
		{
			node->feature_al = NULL;
			node->feature_dt = (JcDetectionFeature*)fxMalloc(sizeof(JcDetectionFeature));
			node->feature_lf = NULL;
			break;
		}
		case LEAF:
		{
			node->feature_al = NULL;
			node->feature_dt = NULL;
			node->feature_lf = (JcLeafFeature*)fxMalloc(sizeof(JcLeafFeature));
			break;
		}
		default:
			break;
	}
	node->left_child = NULL;
	node->right_child = NULL;
	return node;

}
void jcFreeTree(JcTreeNode * root){
	switch (root->node_type){
		case LEAF:{
			fxFree(&(root->feature_lf));
			fxFree(&root);
			break;
		}
		case DETECTION:{
			jcFreeTree(root->left_child);
			jcFreeTree(root->right_child);
			fxFree(&(root->feature_dt));
			fxFree(&root);
			break;
		}
		case ALIGNMENT:{
			jcFreeTree(root->left_child);
			jcFreeTree(root->right_child);
			fxFree(&(root->feature_al));
			fxFree(&root);
			break;
		}

	}
}

JcSampleSet* jcSampleSet(JcTrainData * train_data, int position){
	JcSampleSet* sampleset = (JcSampleSet*)fxMalloc(sizeof(JcSampleSet));

	int pos_total_num = train_data->positive_num;
	int neg_total_num = train_data->negative_num;
	int pos_num = 0;//positive sample in this node
	int neg_num = 0;//negative sample in this node
	int loop_i = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : pos_num)
#endif
	for (loop_i = 0; loop_i < pos_total_num; loop_i++){
		if (train_data->positive_sample[loop_i].Position_in_Tree == position)
			pos_num++;
	}
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : neg_num)
#endif
	for (loop_i = 0; loop_i < neg_total_num; loop_i++){
		if (train_data->negative_sample[loop_i].Position_in_Tree == position&&
			train_data->negative_sample[loop_i].isAvailable)
			neg_num++;
	}
	sampleset->neg_size = neg_num;
	sampleset->pos_size = pos_num;
	sampleset->neg_positions = (int *)fxMalloc(sizeof(int)*neg_num);
	sampleset->pos_positions = (int *)fxMalloc(sizeof(int)*pos_num);


	for (int i = 0,j=0; i < pos_total_num; i++){
		
		if (train_data->positive_sample[i].Position_in_Tree == position){
			(sampleset->pos_positions)[j] = i;
			j++;
		}
	}

	for (int i = 0, j = 0; i < neg_total_num; i++){
		if (train_data->negative_sample[i].isAvailable&&
			train_data->negative_sample[i].Position_in_Tree == position){
				(sampleset->neg_positions)[j] = i;
				j++;
			
		}
	}
	return sampleset;
}

void jcFreeSampleSet(JcSampleSet** sampleset){
	if ((*sampleset)->neg_positions!=NULL)
		fxFree(&((*sampleset)->neg_positions));
	if ((*sampleset)->pos_positions != NULL)
		fxFree(&((*sampleset)->pos_positions));
	fxFree(sampleset);
}


double jcClassificationScore(JcTrainData *train_data,JcSampleSet *sample_set){
	double weight_pos = 0.0;
	double weight_neg = 0.0;
	int pos_num = sample_set->pos_size;
	int neg_num = sample_set->neg_size;
	
	for (int i = 0; i < pos_num; i++){
		int sample_num = (sample_set->pos_positions)[i];
		weight_pos += (train_data->positive_sample[sample_num]).Sample_Weight;
	}

	for (int i = 0; i < neg_num; i++){
		int sample_num = (sample_set->neg_positions)[i];
		weight_neg += (train_data->negative_sample[sample_num]).Sample_Weight;
	}
	if (neg_num == 0)
		weight_neg += 1;// non_zero
	if (pos_num == 0)
		weight_pos += 1;// non_zero
	return 0.5*log(weight_pos / weight_neg);// non_zero
}


/*void jcLeafOffset(JcTrainData *train_data, JcSampleSet * sample_set, FxPoint64 * offset){
	int pos_num = sample_set->pos_size;
	if (pos_num == 0){
		for (int i = 0; i < LANDMARKTYPE; i++){
			offset[i].x = 0.0;
			offset[i].y = 0.0;
		}
	}
	else {
		
		for (int j = 0; j < LANDMARKTYPE; j++){
			double offset_x = 0.0;
			double offset_y = 0.0;
			for (int i = 0; i < pos_num; i++){
				int sample_num = (sample_set->pos_positions)[i];
				JcTrainSample *sample = &(train_data->positive_sample[sample_num]);
				offset_x += ((((sample->GroundTruth)->LandMark)[j]).x - ((sample->Face_Shap_Delta)->LandMark)[j].x);
				offset_y += ((( sample->GroundTruth)->LandMark)[j].y - ((sample->Face_Shap_Delta)->LandMark)[j].y);
			}
			offset[j].x = (double)(offset_x / pos_num);
			offset[j].y = (double)(offset_y / pos_num);
		}
	}
}*/


void jcUpdateTrainSample(JcTrainData* train_data,JcSampleSet *sampleset,double class_score){
	//update classification score
	for (int i = 0; i < sampleset->pos_size; i++){
		int sample_num = sampleset->pos_positions[i];
		JcTrainSample* trainsample = &(train_data->positive_sample[sample_num]);
		trainsample->Class_Score += class_score;
	}

	for (int i = 0; i < sampleset->neg_size; i++){
		int sample_num = sampleset->neg_positions[i];
		JcTrainSample* trainsample = &(train_data->negative_sample[sample_num]);
		trainsample->Class_Score += class_score;	
	}
}





