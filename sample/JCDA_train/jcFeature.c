#include"jcFeature.h"

JcAlignmentFeature* jcCreateAF_Array(uint32 size){
	JcAlignmentFeature * af = fxMalloc(sizeof(JcAlignmentFeature)*size);
	return af;
}
JcDetectionFeature* jcCreateDF_Array(uint32 size){
	JcDetectionFeature * df = fxMalloc(sizeof(JcDetectionFeature)*size);
	return df;
}
void jcFreeAF_Array(JcAlignmentFeature **af){
	fxFree(af);
	*af = NULL;
}
void jcFreeDF_Array(JcDetectionFeature **df){
	fxFree(df);
	*df = NULL;
}
#define RADIUS_1 8
uint32 jcA_DFeatureRadius(int cascade_stage){
	//return  RADIUS_1;
	return (STAGES_OF_CASCADE - cascade_stage)*(STAGES_OF_CASCADE - cascade_stage) + 10;
	//regression or classifier feature random radius 
	//the larger cascade_stage the less radius
	//20 is decide by sample size
}

void jcRandAlFeature(JcAlignmentFeature * a_f, uint32 size, uint32 radius){

	int loop_i = 0;
#ifdef _OPENMP
# pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < size; loop_i++){
		(a_f[loop_i]).off_set[0].x = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		(a_f[loop_i]).off_set[0].y = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		(a_f[loop_i]).off_set[1].x = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		(a_f[loop_i]).off_set[1].y = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
	}

}

void jcRandDeFeature(JcDetectionFeature * d_f, uint32 size, uint32 radius){
	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < size; loop_i++){
		d_f[loop_i].FacialPoint[0] = fxRandom(LANDMARKTYPE, FX_RANDOM_MEAN_HALF_N);
		d_f[loop_i].FacialPoint[1] = fxRandom(LANDMARKTYPE, FX_RANDOM_MEAN_HALF_N);
		//while (d_f[loop_i].FacialPoint[0] == d_f[loop_i].FacialPoint[1])
		//	d_f[loop_i].FacialPoint[1] = fxRandom(LANDMARKTYPE, FX_RANDOM_MEAN_HALF_N);
		d_f[loop_i].off_set[0].x = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		d_f[loop_i].off_set[0].y = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		d_f[loop_i].off_set[1].x = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
		d_f[loop_i].off_set[1].y = fxRandom(radius * 2, FX_RANDOM_MEAN_0);
	}

}