#include "faLocal.h"


void faLocalTree(FaTreeNode** root, int depth,FaTrainData *traindata, int landmark_num,int node_position,int stage_num){
	if (depth < TREE_DEPTH){
		*root = (FaTreeNode*)fxMalloc(sizeof(FaTreeNode));
		(*root)->left_child = NULL;
		(*root)->right_child = NULL;
		(*root)->node_type = INTERNAL;
		//(*root)->node_num = node_position;
		(*root)->feature = faSID(traindata, node_position, landmark_num, stage_num);
		/***********************************************************************/
		
		//printf("Node_position:%d         threshold:%d\n     offset0:(%d,%d)  offset1:(%d,%d) \n", 
		//	node_position, (*root)->feature.threshold,
		//	(*root)->feature.off_set[0].x , (*root)->feature.off_set[0].y,
		//	(*root)->feature.off_set[1].x , (*root)->feature.off_set[1].y);
		//printf("========================================================================\n");
		/***********************************************************************/
		faLocalTree(&((*root)->left_child), depth + 1, traindata, landmark_num, 
			LEFT_CHILD(node_position), stage_num);
		faLocalTree(&((*root)->right_child), depth + 1, traindata, landmark_num,
			RIGHT_CHILD(node_position), stage_num);
	}
	else{
		*root = (FaTreeNode*)fxMalloc(sizeof(FaTreeNode));
		(*root)->left_child = NULL;
		(*root)->right_child = NULL;
		(*root)->node_type = LEAF;
	}
}

void faFreeTree(FaTreeNode** root){

	if (*root != NULL){
		faFreeTree(&((*root)->left_child));
		faFreeTree(&((*root)->right_child));
		fxFree(root);
	}

}