#include "Fa3000.hpp"
#include <memory>




FxAlib::FxAlib()
{
	imp = new FxAlib_Imp;
}

FxAlib::~FxAlib()
{
	delete imp;

}

void FxAlib::cal(FxImageData *ImageData)
{
	image = ImageData;
	imp->alib = this;
	imp->Calc();

}
void FxAlib::setCalcDevice(FxCalcDevice device)
{
	CalcDevice = device;
}

//
//
//
//
//
//
//
//



FxAlib_Imp::FxAlib_Imp()
{
}

void FxAlib_Imp::Calc()
{
	image = alib->image;
	memcpy(LandMark_Clc, Mean_Shape, sizeof(FxPoint64)*LANDMARK_TYPE);
	for (int i = 0; i < STAGE; i++) 
	{
		for (int j = 0; j < TREE_NUM_EACH_STAGE; j++) 
		{
			int landmark = j % LANDMARK_TYPE;
			FxTree* tree = &(((FxForest*)forest)->tree[i][j]);
			int lbf_num = fxLBF(tree, landmark);
			lbf_num += j*TREE_LEAF_NUM;
			fxUpdataStageShape(lbf_num, i);
		}
	}
	for (int i = 0; i < LANDMARK_TYPE; i++)
	{
		alib->LandMark[i].x = (int)LandMark_Clc[i].x;
		alib->LandMark[i].y = (int)LandMark_Clc[i].y;
	}
}

void FxAlib_Imp::fxUpdataStageShape(int lbf_num, int stage)
{
	for (int i = 0; i < LANDMARK_TYPE; i++) 
	{
		LandMark_Clc[i].x += W[stage][i * 2][lbf_num];
		LandMark_Clc[i].y += W[stage][i * 2 + 1][lbf_num];
	}
}

int FxAlib_Imp::fxLBF(FxTree * tree, int landmark_num)
{
	int node_position = 0;
	
	for (int i = 0; i < TREE_DEPTH; i++) 
	{
		FxPoint * offset = tree->sid[node_position].offset;
		int feature = fxFeature(LandMark_Clc[landmark_num], offset);
		if (feature < tree->sid[node_position].threshold) 
		{
			node_position = LEFT_CHILD(node_position);
		}
		else 
		{
			node_position = RIGHT_CHILD(node_position);
		}
	}
	return node_position - (1 << (TREE_DEPTH)) + 1;

}

int FxAlib_Imp::fxFeature(FxPoint64 center, FxPoint * off_set)
{
	FxPoint offset1;
	offset1.x = (int)(center.x + (double)off_set[0].x);
	offset1.y = (int)(center.y + (double)off_set[0].y);
	FxPoint offset2;
	offset2.x = (int)(center.x + (double)off_set[1].x);
	offset2.y = (int)(center.y + (double)off_set[1].y);
	int width = image->width;
	int height = image->height;
	offset1.x = offset1.x < 0 ? 0 : (offset1.x >= width ? width - 1 : offset1.x);
	offset2.x = offset2.x < 0 ? 0 : (offset2.x >= width ? width - 1 : offset2.x);
	offset1.y = offset1.y < 0 ? 0 : (offset1.y >= height ? height - 1 : offset1.y);
	offset2.y = offset2.y < 0 ? 0 : (offset2.y >= height ? height - 1 : offset2.y);
	double a = fxGetRealData(offset1.x, offset1.y);
	double b = fxGetRealData(offset2.x, offset2.y);
	return (int)(a-b);
}

inline double FxAlib_Imp::fxGetRealData(int x, int y)
{
	double value = (double)((unsigned char *)image->data)[image->width*y + x];
	return value;
}

