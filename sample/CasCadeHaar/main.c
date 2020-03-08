#include "Haar.h"
#include "AdaBoost.h"
#include "fxbase.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cv.h"
#include "highgui.h"
#include <time.h>



void readTrainData(AdaBoostTrainData * data){
	FILE *file = fopen("D:\\DATA\\adaboost_test.txt", "r");
	int datasize = 0;
	fscanf(file, "%d\n", &datasize);
	data->DataSize = datasize;
	data->data = (int*)malloc(sizeof(int)*datasize);
	data->label = (char*)malloc(sizeof(char)*datasize);
	for (int i = 0; i < datasize; i++){
		fscanf(file, "%d %d\n", &(data->data[i]), &(data->label[i]));
	}
	fclose(file);

}
int main() {
	/*AdaBoostTrainData traindata;
	clock_t start = clock();
	printf("reading train data\n");
    readTrainData(&traindata);
	for (int i = 0; i < 10; i++) {
		printf("%d,%d\n",(traindata.data[i]),(traindata.label[i]));
	}
	AdaBoostClassifier classifier = AdaBoost(traindata, 10);
	printf("++++++++++++++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < classifier.ClassifierSizeType0; i++) {
		printf("%d ,%lf \n", classifier.ClassifierType0[i],classifier.ClassifierWeight0[i]);
	
	}
	printf("++++++++++++++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < classifier.ClassifierSizeType1; i++) {
		printf("%d ,%lf \n", classifier.ClassifierType1[i], classifier.ClassifierWeight1[i]);

	}

	clock_t end = clock();
	printf("using time:%lf s\n",(double)(end-start)/CLOCKS_PER_SEC);
	
	ReleaseWeakClassifier(classifier);
	system("pause");
	
    getHaar("D:\\Faces_Database\\Face\\",18000,10000,20,20);
	return 0;
	system("pause");*/
    
    //fxCreateMat(fxSize(-1,-1), FX_32C1);
	//FxMat	*src = readImage("/Users/Tony/Data/3.jpg", 0);
	/*FxMat * src = readImage("D:\\Data\\2.jpg",0);
	IplImage *image = cvLoadImage("D:\\Data\\2.jpg", 0);
	//IplImage *dst = cvCreateImage(cvSize(src->width,src->height), 8, src->type);
	//IplImage *dstimage=setImage(src);
	FxMat *dst = fxCreateMat(fxSize(src->width,src->height),FX_32C1);
	//integralImage unitl test
	fxIntegralImage(src, dst);
	uchar *data = src->data;
	int x=1, y=1;
	int res = 0;
	for (int j = 0; j < src->height && j<=y; j++){
		for (int i = 0; i < src->width && i<=x; i++){
			res += data[j*src->width + i];
			printf("%d\n", data[j*src->width + i]);
			printf("%d\n", (int)cvGetReal2D(image, j, i));
		}
	}
	printf("integral image data  x: %d y : %d is %d\n", x, y, ((uint32*)dst->data)[y*(dst->width)+x]);
	printf("loop data  x: %d y : %d is %d\n", x, y, res);
	//uint32 *data = (uint32)dst->data;
	//fxCopy(src,dst);
	FxHaarFeature haar;
	haar.offset = fxPoint(0, 0);
	haar.size = fxSize(3, 3);
	haar.type = FX_HAAR_TYPE4;
	printf("%d", fxHaar(src, haar));
 	fxReleaseMat(&src);
	IplImage *show = setImage(dst);
	cvNamedWindow("Test", 1);
	cvShowImage("Test", show);
	cvWaitKey(0);
	cvReleaseImage(&dst);*/
	clock_t start, end;
	start = clock();
	fxHaarTrain("D:\\Faces_Database\\Face\\", "D:\\Cascade\\",5000, 5000, fxSize(20, 20) );
	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	int hour = time / (60 * 60);
	int minute = (time - hour*(60 * 60)) / 60;
	printf("using time :%d hour(s) %d minute(s) %lf seconds\n", hour, minute, time - hour*60.0*60.0 - minute*60.0);

	getchar();

}