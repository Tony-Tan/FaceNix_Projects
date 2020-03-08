#include "Haar.h"

void fxIntegralImage(FxMat *src, FxMat *dst){
	
	FX_FUNCTION("fxIntegralImage");


	FX_NULL_POINTER_TEST(src);
	FX_NULL_POINTER_TEST(dst);
	FX_DATA_TYPE_TEST(src->type, FX_8C1);
	FX_DATA_TYPE_TEST(dst->type, FX_32C1);
	FX_MAT_SIZE_TEST(src, dst);


	uchar *src_data = src->data;
	uint32 *dst_data = (uint32*)dst->data;
	
	int width=src->width;
	int height = src->height;

	__BEGIN__
	dst_data[0] = (uint32)src_data[0];

	for (int width_i = 1; width_i < width; width_i++){
		dst_data[width_i] = dst_data[width_i - 1] + src_data[width_i];
	}
	for (int height_i = 1; height_i < height; height_i++){
		
		dst_data[height_i*width] = dst_data[(height_i - 1)*width] + src_data[height_i*width];
		
		
	}
	for (int height_i = 1; height_i < height; height_i++){
		for (int width_i = 1; width_i < width; width_i++){
			dst_data[height_i*width + width_i] = dst_data[height_i*width + width_i - 1] +
				dst_data[(height_i - 1)*width + width_i] - dst_data[(height_i - 1)*width + width_i - 1]+
				src_data[height_i*width+width_i];
		
		}
	}


	__END__


}
//
//
//
//
//
//
//

int fxHaar(FxMat *src, FxHaarFeature HaarFeature){
	FX_FUNCTION("fxHaar");
	FX_POINT_OUT_OF_RANGE_TEST(*src, HaarFeature.offset);
	FX_POINT_OUT_OF_RANGE_TEST(*src, fxOffset(HaarFeature.offset, HaarFeature.size));
	FxMat *IntegralMat = fxCreateMat(fxSize(src->width, src->height), FX_32C1);
	if (src->type == FX_8C1)
		fxIntegralImage(src, IntegralMat);
	else if (src->type==FX_32C1){
		fxCopy(src, IntegralMat);
	}
	else{
		fxError(FX_ERROR_DATA_TYPE_WRONG, FUNCTIONNAME, __FILE__, __LINE__);
	}
	__BEGIN__
	int feature_value = 0;
	switch (HaarFeature.type){
		case FX_HAAR_TYPE1:
		{
            int A=fxGetRealData(IntegralMat, HaarFeature.offset.x, HaarFeature.offset.y);
            int B=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width,
								HaarFeature.offset.y);
            int C=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width, 
								HaarFeature.offset.y+HaarFeature.size.height/2);
            int D=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width, 
								HaarFeature.offset.y+HaarFeature.size.height);
            int E=fxGetRealData(IntegralMat, HaarFeature.offset.x, 
								HaarFeature.offset.y+HaarFeature.size.height);
            int F=fxGetRealData(IntegralMat, HaarFeature.offset.x, 
								HaarFeature.offset.y+HaarFeature.size.height/2);
            feature_value=-A+B-2*C+D-E+2*F;
			break;

		}
		case FX_HAAR_TYPE2:
		{
            int A=fxGetRealData(IntegralMat, HaarFeature.offset.x, HaarFeature.offset.y);
            int B=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width/2,
								HaarFeature.offset.y);
            int C=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width,
								HaarFeature.offset.y);
            int D=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width,
								HaarFeature.offset.y+HaarFeature.size.height);
            int E=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width/2,
								HaarFeature.offset.y+HaarFeature.size.height);
            int F=fxGetRealData(IntegralMat, HaarFeature.offset.x, HaarFeature.offset.y+HaarFeature.size.height);
            feature_value=-A+2*B-C+D-2*E+F;
			break;
		}
        case FX_HAAR_TYPE3:
        {
            int A=fxGetRealData(IntegralMat, HaarFeature.offset.x, HaarFeature.offset.y);
            int B=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width/2,
								HaarFeature.offset.y);
            int C=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width, 
								HaarFeature.offset.y);
            int D=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width, 
								HaarFeature.offset.y+HaarFeature.size.height/2);
            int E=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width, 
								HaarFeature.offset.y+HaarFeature.size.height);
            int F=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width/2, 
								HaarFeature.offset.y+HaarFeature.size.height);
            int G=fxGetRealData(IntegralMat, HaarFeature.offset.x, HaarFeature.offset.y+HaarFeature.size.height);
            int H=fxGetRealData(IntegralMat, HaarFeature.offset.x, HaarFeature.offset.y+HaarFeature.size.height/2);
            int I=fxGetRealData(IntegralMat, HaarFeature.offset.x+HaarFeature.size.width/2, 
								HaarFeature.offset.y+HaarFeature.size.height/2);
            feature_value=-A+2*B-C+2*D-E+2*F-G+2*H-4*I;
            break;
        }
        case FX_HAAR_TYPE4:
        {
            int A=fxGetRealData(IntegralMat, HaarFeature.offset.x, 
				HaarFeature.offset.y);
            int B=fxGetRealData(IntegralMat, HaarFeature.offset.x + HaarFeature.size.width/3,
				HaarFeature.offset.y);
            int C=fxGetRealData(IntegralMat, HaarFeature.offset.x + (HaarFeature.size.width/3)*2, 
				HaarFeature.offset.y);
            int D=fxGetRealData(IntegralMat, HaarFeature.offset.x + HaarFeature.size.width,
				HaarFeature.offset.y);


			int E = fxGetRealData(IntegralMat, HaarFeature.offset.x + HaarFeature.size.width,
				HaarFeature.offset.y + HaarFeature.size.height);
            int F = fxGetRealData(IntegralMat, HaarFeature.offset.x + (HaarFeature.size.width/3)*2,
				HaarFeature.offset.y + HaarFeature.size.height);
			int G = fxGetRealData(IntegralMat, HaarFeature.offset.x + HaarFeature.size.width/3,
				HaarFeature.offset.y + HaarFeature.size.height);
			int H = fxGetRealData(IntegralMat, HaarFeature.offset.x, 
				HaarFeature.offset.y + HaarFeature.size.height);
            feature_value=A-3*B+3*C-D+E-3*F+3*G-H;
            break;
        }
		default:
			break;
	
	}
	__END__
	fxReleaseMat(&IntegralMat);
    return feature_value;

}