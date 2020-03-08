#include "fxBase.h"

//temp code for reading image data from file using OpenCV library
/***********************************************************************************/

void readImage_(IplImage * src, FxMat *dst){
	if (src->depth*src->nChannels / 8 != dst->type){
		printf("read image wrong!\n");
		exit(0);
	}
	uchar *dst_data = dst->data;
	uchar *src_data = src->imageDataOrigin;
	for (int i = 0; i < src->height; i++){
		memcpy(dst_data, src_data, dst->type*dst->width);
		dst_data += dst->width;
		src_data += src->widthStep;
	}
}
void setImage_(FxMat *src, IplImage * dst){
	if (dst->depth*dst->nChannels / 8 != src->type){
		printf("read image wrong!\n");
		exit(0);
	}
	//memcpy(dst->imageData, src->data, src->type*src->width*src->height);
	uchar *dst_data = dst->imageDataOrigin;
	uchar *src_data = src->data;
	for (int i = 0; i < src->height; i++){
		memcpy(dst_data, src_data, src->type*src->width);
		dst_data += dst->widthStep;
		src_data += src->width;
	}



}
#include <stdio.h>
FxMat * readImage(char *name, int type){
	IplImage *image = cvLoadImage(name, type);
	if (type == 0) {
		cvSmooth(image, image, CV_GAUSSIAN, 5, 5, 0, 0);
		cvEqualizeHist(image, image);
	}

	FxMat * mat = fxCreateMat(fxSize(image->width, image->height), image->depth*image->nChannels / 8);
	readImage_(image, mat);
	cvReleaseImage(&image);
	return mat;


}
IplImage * setImage(FxMat * src){
	IplImage *image = cvCreateImage(cvSize(src->width, src->height), 8, src->type);
	setImage_(src, image);
	return image;

}


/***********************************************************************************/
#include <stdio.h>
#include <time.h>
char MESSAGE[200];
int TIME_START;
int TIME_END;
void fxProgressBar(char * message,int time_i, int total){
	FX_FUNCTION("fxProgressBar");
	FX_PROCESS_BAR_PARAM_TEST(time_i, total);
	__BEGIN__
	if (time_i == 0)
		TIME_START = clock();
	double percent = (double)(time_i +1 )/total;
	char progress[22];
	for (int i = 0; i < 20; i++){
		if (i < (percent *20.0))
			progress[i] = '*';
		else
		{
			progress[i] = ' ';
		}
		switch (time_i % 4){
		case 0:progress[20] = '/'; break;
		case 1:progress[20] = '-'; break;
		case 2:progress[20] = '\\'; break;
		case 3:progress[20] = '|'; break;
		}
	}

	progress[21] = '\0';
	sprintf(MESSAGE, "%s Complete:%s |%lf %%%%\r",message,progress, percent * 100);
	printf(MESSAGE);
	if (time_i+1  == total){
		
		TIME_END = clock();
		printf("\n************************************************************\n");
		printf("*************Mission use time : %10lf s************",((double)TIME_END-TIME_START)/CLOCKS_PER_SEC);
		printf("\n************************************************************\n");
	}
	__END__
}




FxSize fxSize(int width,int height){
	FX_FUNCTION("fxSize");
	__BEGIN__
		FxSize size;
		size.width=width;
		size.height=height;
    
	__END__
	return size;
}
FxMat* fxCreateMat(FxSize size,FxMatType type){
	FX_FUNCTION("fxCreateMat");
    FX_SIZE_POSITIVE_TEST(size);
	__BEGIN__
		FxMat *mat = (FxMat *)fxMalloc(sizeof(FxMat));
		mat->height=size.height;
		mat->width=size.width;
		mat->type=type;
		mat->width_step=size.width*type;
		mat->data=(uchar*)malloc(mat->width_step*size.height);
	__END__
    return mat;
}
void * fxMalloc(uint32 size){
	FX_FUNCTION("fxMalloc");
	__BEGIN__
		void * pMalloc = NULL;
		pMalloc = (void *)malloc(size);
		FX_MALLOC_MEMORY_TEST(pMalloc);
		
	__END__
	return pMalloc;
}
void fxFree(void **pointer){
	FX_FUNCTION("fxFree");
	FX_NULL_POINTER_TEST((*pointer));
	__BEGIN__
		free((*pointer));
		(*pointer) = NULL;
	__END__

}
void fxReleaseMat(FxMat** mat){
	FX_FUNCTION("fxRealseMat");
	FX_NULL_POINTER_TEST((*mat));
	__BEGIN__
    fxFree((void**)&(*mat)->data);
	fxFree((void**)mat);
	__END__

}
void fxZero(FxMat *mat){
    FX_FUNCTION("fxZero");
    FX_NULL_POINTER_TEST(mat);
    FX_MAT_DATA_TEST(mat);
    __BEGIN__
    memset(mat->data, 0, mat->width_step*mat->height);
    __END__

}
FxPoint fxOffset(FxPoint point,FxSize size){
	FX_FUNCTION("fxOffset");
	__BEGIN__
	FxPoint offsetpoint;
	offsetpoint.x = point.x + size.width;
	offsetpoint.y = point.y + size.height;
	__END__
	return offsetpoint;
}
void fxCopy(FxMat* src, FxMat* dst){
	FX_FUNCTION("fxCopy");
	FX_NULL_POINTER_TEST(src);
	FX_NULL_POINTER_TEST(dst);
	FX_DATA_TYPE_TEST(src->type, dst->type);
	FX_MAT_SIZE_TEST(src, dst);
	FX_NULL_POINTER_TEST(src);
	__BEGIN__
		memcpy(dst->data, src->data, src->width_step*src->height);
	__END__
}
FxPoint fxPoint(int x, int y){
	FX_FUNCTION("fxPoint");
	__BEGIN__
	FxPoint point;
	point.x = x;
	point.y = y;
	__END__

	return point;
}
double fxGetRealData(FxMat *mat, int x, int y){
	FX_FUNCTION("fxGetRealData");
	FX_NULL_POINTER_TEST(mat);
	FX_MAT_DATA_TEST(mat);
	FX_POINT_OUT_OF_RANGE_TEST(*mat, fxPoint(x, y));
	__BEGIN__
	double value = 0.0;
	switch (mat->type)
	{
	case FX_8C1:
		value =(double)((uchar *)mat->data)[mat->width*y + x];
		break;
	case FX_32C1:
		value = (double)((uint32*)mat->data)[mat->width*y + x];
		break;
	case FX_64C1:
		value = (double)((double *)mat->data)[mat->width*y + x];
		break;
	default:
		fxError(FX_ERROR_DATA_TYPE_WRONG,FUNCTIONNAME,__FILE__,__LINE__);
		break;
	}
	__END__

	return value;
}

int FXRANDOM_ALREADY_INIT = 0;
int fxRandom(int n, FxRandomType type){
	FX_FUNCTION("fxRandom");
	FX_PARAM_NEGATIVE_TEST(n);
	__BEGIN__
	if (!FXRANDOM_ALREADY_INIT){
		srand(time(NULL));
		FXRANDOM_ALREADY_INIT = 1;
	}
	__END__
	switch (type)
	{
	case FX_RANDOM_MEAN_0:
		return (rand() % n)-n/2;
	case FX_RANDOM_MEAN_HALF_N:
		return (rand() % n);
	case FX_RANDOM_MEAN_NEGA_HALF_N:
		return (-rand() % n);
	default:
			break;
	}
	return 0; 
}