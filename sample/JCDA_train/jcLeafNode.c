#include "jcForest.h"




void jcLearnLeafNode(JcTrainData* train_data, int position,int face_point,int tree_num, JcLeafFeature *feature){
	JcSampleSet *sampleset = jcSampleSet(train_data, position);
	feature->class_score = jcClassificationScore(train_data, sampleset);
	jcUpdateTrainSample(train_data, sampleset, feature->class_score);
	for (int i = 0; i < sampleset->pos_size; i++){
		int pos_position = sampleset->pos_positions[i];
	
		(train_data->positive_sample[pos_position]).LBF[tree_num] = (position - (1 << TREE_DEPTH) + 1 + tree_num*(1 << TREE_DEPTH));
		
	}
	for (int i = 0; i < sampleset->neg_size; i++){
		int neg_position = sampleset->neg_positions[i];
		(train_data->negative_sample[neg_position]).LBF[tree_num] = position - (1 << TREE_DEPTH) + 1 + tree_num*(1 << TREE_DEPTH);

	}



#ifdef OURPUTPRINT
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("learn a leaf node ,tree position %d \n", position);
	
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif
#ifdef DEBUG
	fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	fprintf(logfile, "learn a leaf node ,tree position %d \n", position);
	fprintf(logfile, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif

	jcFreeSampleSet(&sampleset);
}