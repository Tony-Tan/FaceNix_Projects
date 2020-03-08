#include "faWrite.h"

/*
struct FaSID_{
FxPoint offset[2];
int threshold;
};//shape index difference


struct FaTree_{
FaSID sid[TREE_NODES_NUM];
};


struct FaForest_{
FaTree tree[STAGE][TREE_NUM_EACH_STAGE];
};
FxPoint64 Mean_Shape[LANDMARK_TYPE];
double W[STAGE][LANDMARK_TYPE * 2][TREE_NUM_EACH_STAGE*TREE_LEAF_NUM];
*/

void faWriteData(FaForest * forest)
{	
	FILE * data_file = fopen("../../../data/data.inc", "w+");
	fprintf(data_file, "int forest[]={");
	for (int loop_stage = 0; loop_stage < STAGE; loop_stage++)
	{
		for (int loop_tree = 0; loop_tree < TREE_NUM_EACH_STAGE; loop_tree++)
		{
			for (int loop_node = 0; loop_node < TREE_NODES_NUM; loop_node++)
			{
				if (loop_node== TREE_NODES_NUM-1
					&& loop_tree == TREE_NUM_EACH_STAGE-1
					&& loop_stage == STAGE-1)
				{
					FaSID* sid_ptr = &(forest->tree[loop_stage][loop_tree].sid[TREE_NODES_NUM - 1]);
					fprintf(data_file, "%d,%d,%d,%d,%d\n"
						, sid_ptr->offset[0].x
						, sid_ptr->offset[0].y
						, sid_ptr->offset[1].x
						, sid_ptr->offset[1].y
						, sid_ptr->threshold);
				}
				else
				{
					FaSID* sid_ptr = &(forest->tree[loop_stage][loop_tree].sid[loop_node]);
					fprintf(data_file, "%d,%d,%d,%d,%d,"
						, sid_ptr->offset[0].x
						, sid_ptr->offset[0].y
						, sid_ptr->offset[1].x
						, sid_ptr->offset[1].y
						, sid_ptr->threshold);
				}
			}
			
		}
		

	}
	fprintf(data_file, "};\n");

	//w
	fprintf(data_file, "double Mean_Shape[]={");
	for (int i = 0; i < LANDMARK_TYPE-1; i++)
	{
		fprintf(data_file, "%lf,%lf,", Mean_Shape[i].x, Mean_Shape[i].y);

	}
	fprintf(data_file, "%lf,%lf", Mean_Shape[LANDMARK_TYPE - 1].x, Mean_Shape[LANDMARK_TYPE - 1].y);
	fprintf(data_file, "};\n");
	//mean_shap
	//double W[STAGE][LANDMARK_TYPE * 2][TREE_NUM_EACH_STAGE*TREE_LEAF_NUM];
	fprintf(data_file, "double W[STAGE][LANDMARK_TYPE * 2][TREE_NUM_EACH_STAGE*TREE_LEAF_NUM]={");
	for (int loop_stage = 0; loop_stage < STAGE; loop_stage++)
	{
		for (int loop_type = 0; loop_type < LANDMARK_TYPE * 2; loop_type++)
		{
			for (int tree_loop = 0; tree_loop < TREE_NUM_EACH_STAGE*TREE_LEAF_NUM; tree_loop++)
			{
				if (loop_stage == STAGE-1
					&& loop_type == LANDMARK_TYPE * 2 - 1
					&& tree_loop == TREE_NUM_EACH_STAGE*TREE_LEAF_NUM - 1)
				{
					fprintf(data_file, "%lf", W[loop_stage][loop_type][tree_loop]);
				}
				else {
					fprintf(data_file, "%lf,", W[loop_stage][loop_type][tree_loop]);
				}
			}

		}

	}
	
	fprintf(data_file, "};\n");






	fclose(data_file);
}