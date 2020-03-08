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

#define LANDMARK_TYPE 51
#define TREE_DEPTH 4
#define LANDMARK_FOREST_SIZE 5
#define TREE_NUM_EACH_STAGE (LANDMARK_TYPE*LANDMARK_FOREST_SIZE)
#define STAGE 10
#define OPENMP_THREAD_NUM 16
#define TREE_NODES_NUM ((1 << (TREE_DEPTH))-1)
#define TREE_LEAF_NUM (1 << (TREE_DEPTH))
typedef struct FaSID_ FaSID;
typedef struct FaTree_ FaTree;
typedef struct FaForest_ FaForest;
typedef struct FaData_ FaData;

extern double Mean_Shape[];
extern double W[STAGE][LANDMARK_TYPE * 2][TREE_NUM_EACH_STAGE*TREE_LEAF_NUM];
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
struct FaData_{
	FxMat * mat;
	FxPoint64 landmark[LANDMARK_TYPE];

};




#endif