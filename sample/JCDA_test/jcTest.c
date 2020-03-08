#include "jcTest.h"
#include <stdio.h>
JcLandMark *jcReadMeanShape(char * filename){
	JcLandMark * mean_shape = (JcLandMark*)fxMalloc(sizeof(JcLandMark));
	FILE *file = fopen(filename, "r");
	for (int i = 0; i < LANDMARKTYPE; i++){
		fscanf(file, "%lf %lf\n", &(mean_shape->LandMark[i].x), &(mean_shape->LandMark[i].y));
	}
	fclose(file);
	return mean_shape;
}
void jcFreeMeanShape(JcLandMark * landmark){
	free(landmark);
}
JcTestSample * jcReadSample(FxMat * src_mat,JcLandMark * face_shape){
	JcTestSample *sample = (JcTestSample*)fxMalloc(sizeof(JcTestSample));
	sample->mat = src_mat; 
	sample->Face_Shape = (JcLandMark *)fxMalloc(sizeof(JcLandMark));
	sample->Class_Score = 0.0;
	
	for (int i = 0; i < LANDMARKTYPE; i++){
		sample->Face_Shape->LandMark[i].x = face_shape->LandMark[i].x;
		sample->Face_Shape->LandMark[i].y = face_shape->LandMark[i].y;
	}
	return sample;
}
void jcFreeSample(JcTestSample ** sample){
	free((*sample)->Face_Shape);
	free((*sample));

}
int jcDetectionFeatureValue(FxMat * mat,FxPoint * roi_left_up, FxPoint * center, FxPoint * off_set){
	int a = MAX_FEATURE_VALUE;
	int b = MAX_FEATURE_VALUE;
	int width = mat->width;
	int height = mat->height;
	int off_set0_x = center[0].x + off_set[0].x;
	int off_set0_y = center[0].y + off_set[0].y;
	int off_set1_x = center[1].x + off_set[1].x;
	int off_set1_y = center[1].y + off_set[1].y;
	if (off_set0_x >= 0 && off_set0_x < width&&
		off_set0_y >= 0 && off_set0_y<height)
		a = (int)fxGetRealData(mat, off_set0_x, off_set0_y);
	if (off_set1_x >= 0 && off_set1_x < width&&
		off_set1_y >= 0 && off_set1_y < height)
		b = (int)fxGetRealData(mat, off_set1_x, off_set1_y);
	return a - b;

}

int jcAlignmentFeatureValue(FxMat * mat, FxPoint* center, FxPoint * off_set){
	int a = MAX_FEATURE_VALUE;
	int b = MAX_FEATURE_VALUE;
	int width = mat->width;
	int height = mat->height;
	int off_set0_x = center->x + off_set[0].x;
	int off_set0_y = center->y + off_set[0].y;
	int off_set1_x = center->x + off_set[1].x;
	int off_set1_y = center->y + off_set[1].y;
	if (off_set0_x >= 0 && off_set0_x < width&&
		off_set0_y >= 0 && off_set0_y<height)
		a = (int)fxGetRealData(mat, off_set0_x, off_set0_y);
	if (off_set1_x >= 0 && off_set1_x < width&&
		off_set1_y >= 0 && off_set1_y < height)
		b = (int)fxGetRealData(mat, off_set1_x, off_set1_y);
	return a - b;
}
JcFace *jcCreateFace(){


}
JcCascadeData *casdata = NULL;
JcLandMark  * landmark = NULL;
void jcInitTest(char *cascade_file_path,char * mean_face_shape){
	if (casdata == NULL)
		casdata = jcReadCascadeData("D:\\DATA\\cascade40\\");
	if (landmark == NULL)
		landmark = jcReadMeanShape(mean_face_shape);
}
void jcReleaseTest(){
	if (casdata != NULL)
		fxFree(&casdata);
	if (landmark != NULL)
		fxFree(&landmark );

}
JcFace* jcTest(FxMat *src){
	
	JcFace *face_list = NULL;
	JcFace ** face_list_temp = &face_list;

	int src_width = src->width;
	int src_height = src->height;
	JcLandMark faceshape;
	JcLandMark faceshape_2_regression;
	//get size 2/SCALE 4/SCALE 6/SCALE 8/SCALE
	//
#define SCALE  1
	double image_rate = 0.0;
	for (int image_i = 1; image_i <= SCALE; image_i ++){
		FxMat *mat_resize = NULL;
		int re_width = 0;
		int re_height = 0;
		image_rate = (double)image_i / (double)SCALE;



		if (image_i != SCALE){
			re_width = (int)src_width*image_rate;
			re_height = (int)src_height*image_rate;
			mat_resize = fxCreateMat(fxSize(re_width,re_height), FX_8C1);
			fxResize(src, mat_resize, FX_INTER_LINEAR);
		}
		else{
			mat_resize = src;
			re_width = src_width;
			re_height = src_height;
		}
		
		
		

		
		for (int height_i = 0; height_i <= re_height - ROI_HEIGHT; height_i += 2){
			for (int width_i = 0; width_i <= re_width - ROI_WIDTH; width_i += 2){

				int isFace = 1;
				for (int land_i = 0; land_i < LANDMARKTYPE; land_i++){
					faceshape.LandMark[land_i].x = landmark->LandMark[land_i].x + width_i;
					faceshape.LandMark[land_i].y = landmark->LandMark[land_i].y + height_i;
					faceshape_2_regression.LandMark[land_i].x = landmark->LandMark[land_i].x + width_i;
					faceshape_2_regression.LandMark[land_i].y = landmark->LandMark[land_i].y + height_i;
				}
				double Class_Score = 0.0;
				//printf("x: %d y :%d \n", width_i, height_i);
	/***********************************************************************************************/
				for (int i = 0; i < STAGES_OF_CASCADE; i++){
					for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
						int landmark_num = casdata->CascadeData[i][j].landmark_num;
						double true_threshold = casdata->CascadeData[i][j].threshold;
						JcTreeNode *root = casdata->CascadeData[i][j].root;
						FxPoint facial_point;
						facial_point.x = (int)faceshape.LandMark[landmark_num].x;
						facial_point.y = (int)faceshape.LandMark[landmark_num].y;

						while (root->node_type != LEAF){
							if (root->node_type == DETECTION){
								int* facial_point = root->feature_dt->FacialPoint;
								int a = MAX_FEATURE_VALUE;
								int b = MAX_FEATURE_VALUE;
								
								//FxPoint point[2];
								
								int x0, y0, x1, y1;
								FxPoint *off_set = root->feature_dt->off_set;
								x0 =(int)(faceshape.LandMark[facial_point[0]].x + off_set[0].x );
								y0 = (int)(faceshape.LandMark[facial_point[0]].y + off_set[0].y );
								x1 = (int)(faceshape.LandMark[facial_point[1]].x + off_set[1].x );
								y1 = (int)(faceshape.LandMark[facial_point[1]].y + off_set[1].y );


								if (x0 >= width_i && x0 < width_i + ROI_WIDTH&&
									y0 >= height_i && y0 < height_i + ROI_HEIGHT)
									a = mat_resize->data[y0*re_width + x0];

								if (x1 >= width_i && x1 < width_i + ROI_WIDTH&&
									y1 >= height_i && y1 < height_i + ROI_HEIGHT)
									b = mat_resize->data[y1*re_width + x1];
								
								int value = a - b;
								int threshold = root->feature_dt->threshold;
								if (value < threshold)
									root = root->left_child;
								else
									root = root->right_child;
							}
							else if (root->node_type == ALIGNMENT){
								//int value = jcAlignmentFeatureValue(sample->mat, &facial_point, root->feature_al->off_set);
								int a = MAX_FEATURE_VALUE;
								int b = MAX_FEATURE_VALUE;
								FxPoint *off_set = root->feature_al->off_set;
								int off_set0_x = facial_point.x + off_set[0].x;
								int off_set0_y = facial_point.y + off_set[0].y;
								int off_set1_x = facial_point.x + off_set[1].x;
								int off_set1_y = facial_point.y + off_set[1].y ;
								if (off_set0_x >= width_i && off_set0_x < width_i + ROI_WIDTH&&
									off_set0_y >= height_i && off_set0_y < height_i + ROI_HEIGHT)
									//a = (int)fxGetRealData(mat, off_set0_x, off_set0_y);
									a = mat_resize->data[off_set0_y*re_width + off_set0_x];
								if (off_set1_x >= width_i && off_set1_x <  width_i + ROI_WIDTH&&
									off_set1_y >= height_i && off_set1_y < height_i + ROI_HEIGHT)
									b = mat_resize->data[off_set1_y*re_width + off_set1_x];
								int value= a - b;
								
								
								
								
								int threshold = root->feature_al->threshold;
								if (value < threshold)
									root = root->left_child;
								else
									root = root->right_child;

							}
						}
						Class_Score += (root->feature_lf->class_score);
						if (Class_Score < true_threshold){
							isFace = 0;
							goto non_face;

						}

						faceshape_2_regression.LandMark[landmark_num].x += root->feature_lf->off_set_array[landmark_num].x;
						faceshape_2_regression.LandMark[landmark_num].y += root->feature_lf->off_set_array[landmark_num].y;


					}
					for (int i = 0; i < LANDMARKTYPE; i++){
						faceshape.LandMark[i].x = faceshape_2_regression.LandMark[i].x;
						faceshape.LandMark[i].y = faceshape_2_regression.LandMark[i].y;
					
					}

				}
				
				if (isFace){
					*face_list_temp = (JcFace *)fxMalloc(sizeof(JcFace));
					(*face_list_temp)->face_center.x = (width_i + ROI_WIDTH / 2) / image_rate;
					(*face_list_temp)->face_center.y = (height_i + ROI_HEIGHT / 2) / image_rate;
					(*face_list_temp)->face_size.width = ROI_WIDTH / image_rate;
					(*face_list_temp)->face_size.height = ROI_HEIGHT / image_rate;
					for (int facepoint_i = 0; facepoint_i < LANDMARKTYPE; facepoint_i++){
						(*face_list_temp)->Face_Shape.LandMark[facepoint_i].x = faceshape.LandMark[facepoint_i].x / image_rate;
						(*face_list_temp)->Face_Shape.LandMark[facepoint_i].y = faceshape.LandMark[facepoint_i].y / image_rate;
					}
					(*face_list_temp)->next = NULL;
					face_list_temp = &((*face_list_temp)->next);
				
				
				}
				non_face:;
			}
		}

		if (image_i != SCALE){
			fxReleaseMat(&mat_resize);
		}
	}
/***********************************************************************************************/
	return face_list;
	
}
