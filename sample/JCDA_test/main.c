#include "jcRead.h"
#include "jcTest.h"
#include <time.h>
int main(){

	jcInitTest("D:\\DATA\\cascade40\\", "D:\\DATA\\cascade40\\mean_shap.txt");

	char *image_path = "D:\\Faces_Database\\FaceNixData40\\face\\";
	//char* image_name="D:\\Faces_Database\\face7.png";
	char image_name[200];
	int has_face = 0;
	for (int i = 0; i < 10000; i++){
		sprintf(image_name, "%s%d%s", image_path, i, ".png");
		FxMat *image = readImage(image_name, 0);
		clock_t start, end;
		start = clock();
		JcFace * face = jcTest(image);
		end = clock();
		printf("using time: %lf ms\n", (double)(end - start)*1000.0 / CLOCKS_PER_SEC);
		IplImage *show = setImage(image);
		JcFace *face_temp = face;
		if (face_temp != NULL){
			has_face++;

			for (int j = 0; j < LANDMARKTYPE; j++){
				cvCircle(show, cvPoint(face_temp->Face_Shape.LandMark[j].x, face_temp->Face_Shape.LandMark[j].y),
					1, CV_RGB(255, 255, 255), 1, 0, 0);

			}
		
		}
		while (face_temp != NULL){
			//cvCircle(show, cvPoint(face_temp->face_center.x, face_temp->face_center.y), face_temp->face_size.width / 2, CV_RGB(255, 255, 255), 2, 0, 0);
			face_temp = face_temp->next;
			printf("%d:face num:%d\n", i, has_face);
			cvNamedWindow("window", 1);
			cvShowImage("window", show);
		
			cvWaitKey(0);
			
		}
		
		
		
			cvWaitKey(10);
		
		fxReleaseMat(&image);
		cvReleaseImage(&show);
		
		
	}
	
	getchar();



}