#include "fa3000test.h"
#include <time.h> 
void update_init_shap(FxPoint64* last_frame,FxPoint64* this_frame,FxPoint64 * target_frame)
{
	for (int i = 0; i < LANDMARK_TYPE; i++) 
	{
		target_frame[i].x = (last_frame[i].x*0.5 + this_frame[i].x*0.5) ;
		target_frame[i].y = (last_frame[i].y*0.5 + this_frame[i].y*0.5) ;
	}

}

int main() {
	//faRead_CreateForest("D:\\Data\\FA3000\\51\\");

	cvNamedWindow("TEST", 1);
	char file_name[512];
	char landmark_file_name[512];
	char file_save_name[512];
	clock_t start, end;


	//camera test 
	IplImage* pFrame = NULL;

	//»ñÈ¡ÉãÏñÍ·  
	CvCapture* pCapture = cvCreateFileCapture("D:\\1.avi");
	FxPoint64 landmark[LANDMARK_TYPE];
	
	FxPoint64 ground_truth[LANDMARK_TYPE];
	//memcpy(landmark, Mean_Shape, sizeof(FxPoint64) * LANDMARK_TYPE);
	double diff = 0.0;
	for (int j=3000;j<3700;j++ ){
	/*while (1)
	{
		pFrame = cvQueryFrame(pCapture);
		IplImage * resize_frame = cvCreateImage(cvSize(128, 128), pFrame->depth, 3);
		cvResize(pFrame, resize_frame, NULL);
		//cvSmooth(resize_frame, resize_frame, CV_GAUSSIAN, 5, 5, 0, 0);
		IplImage * gray_frame = cvCreateImage(cvSize(128, 128), pFrame->depth, 1);

		cvCvtColor(resize_frame, gray_frame, CV_RGB2GRAY);
		cvEqualizeHist(gray_frame, gray_frame);
		FaData* data = faReadCameraData(gray_frame, -1, landmark);

		faTest(data, forest);
		for (int i = 0; i < LANDMARK_TYPE; i++) {
			cvCircle(resize_frame, cvPoint((int)data->landmark[i].x, (int)data->landmark[i].y), 2, CV_RGB(255, 0, 255), 1, 0, 0);

		}
		update_init_shap(Mean_Shape, data->landmark, landmark);

		cvShowImage("TEST", resize_frame);
		if (27 == cvWaitKey(0))
			break;
		faFreeTestData(&data);
		cvReleaseImage(&gray_frame);
		cvReleaseImage(&resize_frame);




		*/




		

		sprintf(file_name, "%s%d%s", "D:\\Data\\51landmark\\face128\\", j , ".png");

		IplImage* image = cvLoadImage(file_name, 1);
		FaData* data = faReadTestData(file_name, 0, landmark);
		start = clock();
		faTest(data);
		end = clock();
		printf("Test time :%d ms\n", end - start);
		for (int i = 0; i < LANDMARK_TYPE; i++)
		{
			cvCircle(image, cvPoint((int)data->landmark[i].x, (int)data->landmark[i].y), 1, CV_RGB(0, 0, 255), 1, 0, 0);

		}
		
		////=============================================================================/
		sprintf(landmark_file_name, "%s%d%s", "D:\\Data\\51landmark\\face128\\", j, ".pts");
		faReadGTLandmark(landmark_file_name, ground_truth);
		diff += faDifference(data->landmark, ground_truth, 128*128);
		///==============================================================================/
		
		cvShowImage("TEST", image);

		sprintf(file_name, "%s%d%s", "D:\\Data\\FA_Image\\51\\", j, ".jpg");
		cvSaveImage(file_name, image,0);
		if(27==cvWaitKey(10))
			break;
		faFreeTestData(&data);
		cvReleaseImage(&image);

		


		


	}
	printf("===================================================\n");
	printf("Difference : %lf \n", diff/700);
	printf("===================================================\n");
	cvWaitKey(0);

}