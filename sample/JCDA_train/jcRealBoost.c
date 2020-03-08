#include "jcRealBoost.h"



void jcUpdateSampleWeight(JcTrainData *train_data){
	int neg_num = train_data->negative_num;
	int pos_num = train_data->positive_num;
	
	int loop_i = 0;


#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < neg_num; loop_i++){
		JcTrainSample* neg_sample = &(train_data->negative_sample[loop_i]);
		if (neg_sample->isAvailable){
			neg_sample->Sample_Weight = exp(-1.0*(neg_sample->Label)*(neg_sample->Class_Score));
		}
		neg_sample->Position_in_Tree = 0;
	}

#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < pos_num; loop_i++){
		JcTrainSample* pos_sample = &(train_data->positive_sample[loop_i]);
		pos_sample->Sample_Weight = exp(-1.0*(pos_sample->Label)*(pos_sample->Class_Score));
		pos_sample->Position_in_Tree = 0;
	}


}