#ifndef FAREAD_H
#define FAREAD_H
#include "faType.h"
#ifdef _OPENMP
#include <omp.h>
#endif


FaForest* faRead_CreateForest(char * file_path);
FaData* faReadTestData(char *file_name, int landmark_init, FxPoint64* landmark);
FaData* faReadCameraData(IplImage *frame, int landmark_init, FxPoint64* landmark);
void faReadGTLandmark(char *landmark_file, FxPoint64* landmark);
double faDifference(FxPoint64 *landmark_calc, FxPoint64 *landmark_gt, int image_size);
void faFreeTestData(FaData** data);
void faFreeForest(FaForest ** forest);
#endif