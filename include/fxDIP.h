#ifndef FXDIP_H
#define FXDIP_H
#include "fxBase.h"
#include "fxTypes.h"

#include "fxError.h"
typedef enum  FxResizeType_ FxResizeType;
enum FxResizeType_{
	FX_INTER_NN,			// ����ھӲ岹
	FX_INTER_LINEAR,		// ˫���Բ�ֵ��Ĭ�Ϸ�����
	FX_INTER_AREA
};
void fxResize(FxMat *src, FxMat *dst, FxResizeType type);
#endif