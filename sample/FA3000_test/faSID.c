//shape index different
#include "faSID.h"
int faFeature(FxMat * mat, FxPoint64 center, FxPoint* off_set/*2 point offset array*/){
	FxPoint offset1;
	offset1.x = (int)(center.x + (double)off_set[0].x);
	offset1.y = (int)(center.y + (double)off_set[0].y);
	FxPoint offset2;
	offset2.x = (int)(center.x + (double)off_set[1].x);
	offset2.y = (int)(center.y + (double)off_set[1].y);
	int width = mat->width;
	int height = mat->height;
	offset1.x = offset1.x < 0 ? 0 : (offset1.x >= width ? width - 1 : offset1.x);
	offset2.x = offset2.x < 0 ? 0 : (offset2.x >= width ? width - 1 : offset2.x);
	offset1.y = offset1.y < 0 ? 0 : (offset1.y >= height ? height - 1 : offset1.y);
	offset2.y = offset2.y < 0 ? 0 : (offset2.y >= height ? height - 1 : offset2.y);
	double a = fxGetRealData(mat, offset1.x, offset1.y);
	double b = fxGetRealData(mat, offset2.x, offset2.y);

	return (int)(a - b);
}
