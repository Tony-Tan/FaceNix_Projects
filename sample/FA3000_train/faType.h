#ifndef FATYPE_H
#define FATYPE_H
#include <stdlib.h>
#include "fxBase.h"
#include "fxTypes.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define LEFT_CHILD(x) ((x)*2+1)
#define RIGHT_CHILD(x) ((x)*2+2)


#define TRAIN_DATA_SIZE 3000
#define LANDMARK_TYPE 51
#define TREE_DEPTH 4
#define LANDMARK_FOREST_SIZE 5
#define TREE_NUM_EACH_STAGE (LANDMARK_TYPE*LANDMARK_FOREST_SIZE)
#define STAGE 10
#define OPENMP_THREAD_NUM 16
#define SID_RANDOM_NUM 500
#define INIT_MEAN_SHAPE_NOICE_RADIUS 10
#define MIN_SID_FEATURE -255
#define MAX_SID_FEATURE 255








typedef unsigned char LBF8;
typedef unsigned int LBF32;
typedef double LBF64;
typedef struct FaData_ FaData;
typedef struct FaTrainData_ FaTrainData;
typedef struct FaSIDFeature_ FaSIDFeature;
struct FaSIDFeature_{
	FxPoint off_set[2];
	int threshold;
};

typedef struct FaTreeNode_ FaTreeNode;
typedef enum Tree_Node_Type_ Tree_Node_Type;

enum Tree_Node_Type_{ LEAF = 0, INTERNAL = 1 };

struct FaTreeNode_{
	//int node_num;/*num label where the node is now,like root=1,root's left child = 1*/
	Tree_Node_Type node_type;
	FaSIDFeature feature;
	FaTreeNode * left_child;
	FaTreeNode * right_child;
};
struct FaData_{
	int tree_position;
	FxMat* image;
	FxPoint64  landmark_tar[LANDMARK_TYPE];
	FxPoint64  landmark_delta[LANDMARK_TYPE];
	FxPoint64  landmark_realtime[LANDMARK_TYPE];
#if TREE_DEPTH<8
	LBF8 lbf[TREE_NUM_EACH_STAGE];
#elif TREE_DEPTH<32
	LBF32 lbf[TREE_NUM_EACH_STAGE];
#elif TREE_DEPTH<64
	LBF64 lbf[TREE_NUM_EACH_STAGE];
#endif
};


struct FaTrainData_{
	FaData* data_array;
	int data_size;
};

#endif