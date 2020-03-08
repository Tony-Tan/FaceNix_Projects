#ifndef FX_BASE_H
#define FX_BASE_H

#include "fxTypes.h"
#include "fxError.h"

/******************************************************************************/
#include "highgui.h"
#include "cv.h"
void readImage_(IplImage * src, FxMat *dst);
void setImage_(FxMat *src, IplImage * dst);


FxMat * readImage(char *name, int type);
IplImage * setImage(FxMat *src);
/******************************************************************************/
void fxProgressBar(char * message,int time_i, int total);
FxSize fxSize(int width,int height);
FxPoint fxPoint(int x, int y);
FxMat* fxCreateMat(FxSize size,FxMatType type);
void * fxMalloc(uint32 size);
void fxReleaseMat(FxMat** mat_pointer);
void fxFree(void **pointer);
FxPoint fxOffset(FxPoint point, FxSize size);
void fxCopy(FxMat* src,FxMat* dst);
double fxGetRealData(FxMat *mat, int x, int y);
int fxRandom(int n,FxRandomType type);
#endif