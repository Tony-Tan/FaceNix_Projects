#include "FxAlib.hpp"
#include <highgui.h>
#include <time.h>


int main()
{
	//faRead_CreateForest("D:\\Data\\FA3000\\51\\");
	FxImageData image;
	image.width = 128;
	image.height = 128;
	image.data = (unsigned char*)malloc(sizeof(unsigned char) * 128 * 128);


	cvNamedWindow("TEST", 1);
	char file_name[512];
	clock_t start, end;
	FxAlib alib;
	for (int j = 3000; j < 3700; j++) 
	{

		sprintf(file_name, "%s%d%s", "D:\\Data\\51landmark\\face128\\", j, ".png");

		IplImage* img = cvLoadImage(file_name, 0);
		for (int h = 0; h < 128; h++) 
		{
			for (int w = 0; w < 128; w++)
			{
				image.data[w + h * 128] = (unsigned char)cvGetReal2D(img, h,w);
			}
			
		}
		
		start = clock();
		alib.cal(&image);
		end = clock();
		printf("Test time :%d ms\n", end - start);
		for (int i = 0; i < LANDMARK_TYPE; i++)
		{
			cvCircle(img, cvPoint(alib.LandMark[i].x, alib.LandMark[i].y), 1, CV_RGB(0, 0, 255), 1, 0, 0);
		}
		cvShowImage("TEST", img);
		sprintf(file_name, "%s%d%s", "D:\\Data\\FA_Image\\51\\", j, ".png");
		cvSaveImage(file_name, img, 0);
		if (27 == cvWaitKey(10))
			break;
		cvReleaseImage(&img);

	}
	free(image.data);

}
