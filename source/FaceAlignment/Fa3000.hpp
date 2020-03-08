#ifndef SRC_Fx_Fx_3000_HPP_
#define SRC_Fx_Fx_3000_HPP_
#include <stdlib.h>
#include "FxAlib.hpp"


#define LANDMARK_TYPE 51
#define TREE_DEPTH 4
#define LANDMARK_FOREST_SIZE 5
#define TREE_NUM_EACH_STAGE (LANDMARK_TYPE*LANDMARK_FOREST_SIZE)
#define STAGE 10
#define TREE_NODES_NUM ((1 << (TREE_DEPTH))-1)
#define TREE_LEAF_NUM (1 << (TREE_DEPTH))
#define LEFT_CHILD(x) ((x)*2+1)
#define RIGHT_CHILD(x) ((x)*2+2)
#include "../../data/data.inc"
typedef struct FxSID_ 
{
	FxPoint offset[2];
	int threshold;
}FxSID;//shape index difference


typedef struct FxTree_
{
	FxSID sid[TREE_NODES_NUM];
}FxTree;

typedef struct FxForest_
{
	FxTree tree[STAGE][TREE_NUM_EACH_STAGE];
}FxForest;

typedef struct FxPoint64_
{
	double x;
	double y;

}FxPoint64;


class FxAlib_Imp
{
public:
	FxAlib_Imp();
	void Calc();
public:
	FxAlib  * alib;
private:
	void fxUpdataStageShape(int lbf_num, int stage);
	int fxLBF(FxTree* tree,int landmark_num);
	int fxFeature(FxPoint64 center, FxPoint* off_set/*2 point offset array*/);
	inline double fxGetRealData(int x, int y);

private:
	FxPoint64 LandMark_Clc[LANDMARK_TYPE];
	FxImageData* image;
};
#endif

