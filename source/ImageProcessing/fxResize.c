#include "fxDIP__.h"


void fxResize(FxMat *src, FxMat *dst, FxResizeType type){
	FX_FUNCTION("fxResize");
	FX_DATA_TYPE_TEST(src->type, dst->type);
	FX_MAT_DATA_TEST(src);
	FX_MAT_DATA_TEST(dst);
	__BEGIN__
	double width_src_d = (double)src->width;
	double height_src_d = (double)src->height;
	double width_dst_d = (double)dst->width;
	double height_dst_d = (double)dst->height;
	double rate_x = width_dst_d / width_src_d;
	double rate_y = height_dst_d / height_src_d;
	int width_dst = dst->width;
	int height_dst = dst->height;
	int width_src = src->width;
	switch (type)
	{
	case FX_INTER_LINEAR:{
		if (src->type == FX_8C1){
			uchar * data_dst = (uchar* )dst->data;
			uchar * data_src = (uchar *)src->data;
			for (int j = 0; j < height_dst-1; j++){
				for (int i = 0; i < width_dst-1; i++){
					double x_src_d = ((double)i) / rate_x;
					double y_src_d = ((double)j) / rate_y;
					int x_src_i = (int)x_src_d;
					int y_src_i = (int)y_src_d;
					double pixel_rate_x = (x_src_d - x_src_i);
					double pixel_rate_y = (y_src_d - y_src_i);
					data_dst[j*width_dst + i] =
					(1.0 - pixel_rate_y)*
					(pixel_rate_x			*	(data_src[y_src_i*width_src + x_src_i+1])		+
					(1.0 - pixel_rate_x)	*	(data_src[y_src_i*width_src + x_src_i]))		+
					pixel_rate_y*
					(pixel_rate_x			*	data_src[(y_src_i + 1)*width_src + x_src_i+1]	+
					(1.0 - pixel_rate_x)	*	data_src[(y_src_i + 1)*width_src + x_src_i ])	;
				}
			}
		}
		else if (src->type == FX_32C1){
			uint32 * data_dst = (uint32 *)dst->data;
			uint32 * data_src = (uint32 *)src->data;
			for (int j = 0; j < height_dst; j++){
				for (int i = 0; i < width_dst; i++){
					double x_src_d = i / rate_x;
					double y_src_d = j / rate_y;
					int x_src_i = (int)x_src_d;
					int y_src_i = (int)y_src_d;
					double pixel_rate_x = (x_src_d - x_src_i);
					double pixel_rate_y = (y_src_d - y_src_i);
					data_dst[j*width_dst + i] =
						pixel_rate_y*(pixel_rate_x*(data_src[y_src_i*width_src + x_src_i]) +
						(1.0 - pixel_rate_x)*(data_src[y_src_i*width_src + (x_src_i + 1)])) +
						(1.0 - pixel_rate_y)*(pixel_rate_x*data_src[(y_src_i + 1)*width_src + x_src_i] +
						(1.0 - pixel_rate_x)*data_src[(y_src_i + 1)*width_src + x_src_i + 1]);
				}
			}
		}
		else if (src->type == FX_64C1){
			uint64 * data_dst = (uint64 *)dst->data;
			uint64 * data_src = (uint64 *)src->data;
			for (int j = 0; j < height_dst; j++){
				for (int i = 0; i < width_dst; i++){
					double x_src_d = i / rate_x;
					double y_src_d = j / rate_y;
					int x_src_i = (int)x_src_d;
					int y_src_i = (int)y_src_d;
					double pixel_rate_x = (x_src_d - x_src_i);
					double pixel_rate_y = (y_src_d - y_src_i);
					data_dst[j*width_dst + i] =
						pixel_rate_y*(pixel_rate_x*(data_src[y_src_i*width_src + x_src_i]) +
						(1.0 - pixel_rate_x)*(data_src[y_src_i*width_src + (x_src_i + 1)])) +
						(1.0 - pixel_rate_y)*(pixel_rate_x*data_src[(y_src_i + 1)*width_src + x_src_i] +
						(1.0 - pixel_rate_x)*data_src[(y_src_i + 1)*width_src + x_src_i + 1]);
				}
			}
		}

	
		break;
	}
	case FX_INTER_NN:{
		if (src->type == FX_8C1){
			uchar * data_dst = (uchar*)dst->data;
			uchar * data_src = (uchar *)src->data;
			for (int j = 0; j < height_dst; j++){
				for (int i = 0; i < width_dst; i++){
					int x_src_i = (int)(double)i / rate_x;
					int y_src_i = (int)(double)j / rate_y;
					data_dst[j*width_dst + i] = data_src[y_src_i*width_src + x_src_i];
				}
			}
		 } else if (src->type == FX_32C1){
			uint32 * data_dst = (uint32 *)dst->data;
			uint32 * data_src = (uint32 *)src->data;
			for (int j = 0; j < height_dst; j++){
				for (int i = 0; i < width_dst; i++){
					int x_src_i = (int)(double)i / rate_x;
					int y_src_i = (int)(double)j / rate_y;
					data_dst[j*width_dst + i] = data_src[y_src_i*width_src + x_src_i];

				}
			 }
		}else if (src->type == FX_64C1){
			uint64 * data_dst = (uint64*)dst->data;
			uint64 * data_src = (uint64 *)src->data;
			for (int j = 0; j < height_dst; j++){
				for (int i = 0; i < width_dst; i++){
					int x_src_i = (int)(double)i / rate_x;
					int y_src_i = (int)(double)j / rate_y;
					data_dst[j*width_dst + i] = data_src[y_src_i*width_src + x_src_i];

				 }
			}
		}


		break;
		}
	case FX_INTER_AREA:{
		
				break;
	}
		default:
			break;
	}
	
	
	__END__

}