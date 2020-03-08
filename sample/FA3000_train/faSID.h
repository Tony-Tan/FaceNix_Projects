#ifndef FASID_H
#define FASID_H
#include "faType.h"
typedef struct FaDataInNode_ FaDataInNode;
struct FaDataInNode_{
	FaData** fadata_p_array;
	int fadata_size;
};
FaSIDFeature faSID(FaTrainData *traindata, int tree_position, int landmark_num, int stage_num);
FaDataInNode * faDataInNode(FaTrainData *traindata, int position);
void faFreeDataInNode(FaDataInNode ** fadata_innode);
#endif