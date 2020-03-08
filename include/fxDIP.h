#ifndef FXDIP_H
#define FXDIP_H
#include "fxBase.h"
#include "fxTypes.h"

#include "fxError.h"
typedef enum  FxResizeType_ FxResizeType;
enum FxResizeType_{
	FX_INTER_NN,			// 最近邻居插补
	FX_INTER_LINEAR,		// 双线性插值（默认方法）
	FX_INTER_AREA
};
void fxResize(FxMat *src, FxMat *dst, FxResizeType type);
#endif