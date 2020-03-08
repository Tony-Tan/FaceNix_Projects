#include "faRead.h"
#define PTS_EX ".pts"






void faReadTree_(FaTree* tree,FILE* file,int position){
	if (position < (1 << (TREE_DEPTH+1))){
		int landmark, node_type;
		fscanf(file, "%d :%d\n", &landmark, &node_type);
		if (node_type == 1){
			fscanf(file, "%d %d ", &(tree->sid[position].offset[0].x), &(tree->sid[position].offset[0].y));
			fscanf(file, "%d %d ", &(tree->sid[position].offset[1].x), &(tree->sid[position].offset[1].y));
			fscanf(file, "%d\n", &(tree->sid[position].threshold));
			faReadTree_(tree, file, LEFT_CHILD(position));
			faReadTree_(tree, file, RIGHT_CHILD(position));
		}
		else if (node_type == 0){
			fscanf(file, "\n");
		}
	}
	
}

void faReadTree(FaTree* tree, FILE* file, int position){
	faReadTree_(tree, file, position);
	fscanf(file, "\n\n");

}


FaForest* faRead_CreateForest(char * file_path) {
}

void faFreeForest(FaForest ** forest){
	fxFree(forest);

}
extern double  Mean_Shape[];
FaData* faReadTestData(char *file_name,int landmark_init,FxPoint64* landmark){
	FaData * data = (FaData*)fxMalloc(sizeof(FaData));
	data->mat = readImage(file_name, 0);
	if (landmark_init == 0)
	{
		for (int i = 0; i < LANDMARK_TYPE; i++) 
		{
			data->landmark[i].x = ((FxPoint64*) Mean_Shape)[i].x;
			data->landmark[i].y = ((FxPoint64*) Mean_Shape)[i].y;
		}
	}
	else if (landmark_init == -1)
	{
		for (int i = 0; i < LANDMARK_TYPE; i++) 
		{
			data->landmark[i].x = landmark[i].x;
			data->landmark[i].y = landmark[i].y;
		}

	}
	return data;
}

FaData* faReadCameraData(IplImage *frame , int landmark_init, FxPoint64* landmark) {
	FxPoint64 * MEAN_SHAPE = &(Mean_Shape[0]);
	FaData * data = (FaData*)fxMalloc(sizeof(FaData));
	data->mat = fxCreateMat(fxSize(frame->width,frame->height),FX_8C1);
	readImage_(frame, data->mat);
	if (landmark_init == 0)
	{
		for (int i = 0; i < LANDMARK_TYPE; i++)
		{
			data->landmark[i].x = MEAN_SHAPE[i].x ;
			data->landmark[i].y = MEAN_SHAPE[i].y ;
		}
	}
	else if (landmark_init == -1)
	{
		for (int i = 0; i < LANDMARK_TYPE; i++)
		{
			data->landmark[i].x = landmark[i].x;
			data->landmark[i].y = landmark[i].y;
		}

	}
	return data;
}

void faFreeTestData(FaData** data){
	fxReleaseMat(&((*data)->mat));
	free((*data));
	*data = NULL;
}

void faReadGTLandmark(char *landmark_file, FxPoint64* landmark) {
	FILE* file = fopen(landmark_file, "r");
	double temp_x, temp_y;
	for (int i = 0; i < LANDMARK_TYPE; i++) {
		fscanf(file, "%lf %lf\n", &temp_x, &temp_y);
		landmark[i].x = temp_x;
		landmark[i].y = temp_y;
	}
	fclose(file);

}
double faDifference(FxPoint64 *landmark_calc, FxPoint64 *landmark_gt,int image_size) {
	double diff = 0.0;
	for (int i = 0; i < LANDMARK_TYPE; i++)
	{
		diff += (landmark_calc[i].x - landmark_gt[i].x)*(landmark_calc[i].x - landmark_gt[i].x);
		diff += (landmark_calc[i].y - landmark_gt[i].y)*(landmark_calc[i].y - landmark_gt[i].y);

	}
	diff /= image_size;
	return diff;
}