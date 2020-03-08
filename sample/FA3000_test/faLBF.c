#include "faLBF.h"
int faLBF(FaData* data, FaTree* tree,int landmark_num){
	int node_position = 0;
	FxMat * mat = data->mat;
	FxPoint64 * landmark = data->landmark;
	for (int i = 0; i < TREE_DEPTH; i++){
		FxPoint * offset = tree->sid[node_position].offset;
		int feature = faFeature(mat, landmark[landmark_num], offset);
		if (feature < tree->sid[node_position].threshold){
			node_position = LEFT_CHILD(node_position);
		}
		else{
			node_position = RIGHT_CHILD(node_position);
		}
	}
	return node_position - (1 << (TREE_DEPTH)) + 1;

}