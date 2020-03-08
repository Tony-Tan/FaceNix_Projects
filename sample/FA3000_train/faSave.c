#include "faSave.h"

void faSaveTree_(FILE* file, FaTreeNode *root, int landmark){
	if (root->node_type == INTERNAL){
		fprintf(file, "%d :%d\n", landmark, root->node_type);
		fprintf(file, "%d %d ", root->feature.off_set[0].x, root->feature.off_set[0].y);
		fprintf(file, "%d %d ", root->feature.off_set[1].x, root->feature.off_set[1].y);
		fprintf(file, "%d\n", root->feature.threshold);
		faSaveTree_(file, root->left_child, landmark);
		faSaveTree_(file, root->right_child, landmark);
	}
	else if (root->node_type == LEAF)
		fprintf(file, "%d :%d\n", landmark, root->node_type);
}

void faSaveTree(FILE* file, FaTreeNode *root, int landmark){
	faSaveTree_(file, root, landmark);
	fprintf(file, "\n\n");

}
void faSaveparam(char * path){
	char Param_name[512];
	sprintf(Param_name, "%s%s%s", path, "param", ".fa");
	FILE* file = fopen(Param_name, "w+");
	fprintf(file, "stage:%d\nlandmark_type:%d\ntree_depth:%d\ntrees_per_stage:%d\n", STAGE, LANDMARK_TYPE, TREE_DEPTH, TREE_NUM_EACH_STAGE);
//#define LANDMARK_TYPE 27
//#define TREE_DEPTH 5
//#define LANDMARK_FOREST_SIZE 25
//#define TREE_NUM_EACH_STAGE (LANDMARK_TYPE*LANDMARK_FOREST_SIZE)
//#define STAGE 5
	fclose(file);
}