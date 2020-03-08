#include "AdaBoost.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
static int GetMinData(AdaBoostTrainData data) {
	if (data.DataSize == 0) {
		printf("getMaxData Wrong:DataSize==0!!\n");
		exit(0);
	}
	int mindata = INT_MAX;
	for (int i = 0; i < data.DataSize; i++) {
		mindata = mindata >= data.data[i] ? data.data[i] : mindata;
	}
	if (mindata == INT_MAX) {
		printf("getMinData Wrong:No data or data Wrong!\n");
		exit(0);
	}
	return mindata;
}
static int GetMaxData(AdaBoostTrainData data) {
	if(data.DataSize==0) {
		printf("getMaxData Wrong:DataSize==0!!\n");
		exit(0);
	}

	int maxdata = INT_MIN;
	for (int i = 0; i < data.DataSize; i++) {
		maxdata = maxdata < data.data[i] ? data.data[i] : maxdata;
	}
	if (maxdata == INT_MIN) {
		printf("getMaxData Wrong:No data or data Wrong!\n");
		exit(0);
	}
	return maxdata;
}
static void InitDataWeight(AdaBoostTrainData data,double *weight) {	
	//¡¶Rapid Object Detection using a Boosted Cascade of Simple Features¡· algorithm
	/*
	int PositiveDataSize = 0;
	int NagetiveDataSIze = 0;
	for (int i = 0; i < data.DataSize; i++) {
		if (data.label[i] == 1)
			PositiveDataSize++;
		else
			NagetiveDataSIze++;
	}
	for (int i = 0; i < data.DataSize; i++) {
		if (data.label[i] == 1)
			weight[i] = 1.0 / (2.0*PositiveDataSize);
		else
			weight[i] = 1.0 / (2.0*NagetiveDataSIze);
	}*/
	//Hang Li method
	int i;

	for (i = 0; i < data.DataSize; i++){
		weight[i] = (double)1.0 / (double)data.DataSize;
	}

}


void NormalizeWeight(double *weight,int size) {
	double weight_sum = 0.0;
	int i;
	for (i = 0; i < size; i++) {
		weight_sum += weight[i];
	}
	if (weight_sum == 0.0) {
		printf("NormalizeWeight Wrong:weight_sum==0!!\n");
		exit(0);
	}
	for (i = 0; i < size; i++) {
		weight[i] /= weight_sum;
		
		//printf("%d %lf \n", i, weight[i]);
	}
	//printf("------------------------------\n");
}

void GetBestAtomClassifier(AdaBoostTrainData data, AdaBoostAtomClassifier * atomclassifier, double * weight) {
	int mindata = GetMinData(data);
	int maxdata = GetMaxData(data);
	AdaBoostAtomClassifier temp;
	temp.error = DBL_MAX;
	//------------------------------------------------------------
	//printf("weight before class:\n");
	//for (int i = 0; i < data.DataSize; i++)
	//	printf("%d,%lf \n", data.data[i], weight[i]);
	//printf("weight after class:\n");
	//------------------------------------------------------------
	if (maxdata - mindata < data.DataSize){
		for (int i = mindata; i <= maxdata; i++) {
			double error_type0 = 0.0;
			double error_type1 = 0.0;
			/*********************************************************/
			// 1, 1, 1, 1, 0, i, 1, 0 , 0, 0, 0
			// f  f  f  f  t  x  t  f   f  f  f
			// t: Classification Right; f: Classification wrong;
			//if data >= classifier&&data.label == 1 is right class type = 1
			/*********************************************************/
			
			
			for (int j = 0; j < data.DataSize; j++) {
				if (data.label[j] == 1){
					if (data.data[j] < i)
						error_type1 += weight[j];
					else
						error_type0 += weight[j];

				}
				else if (data.label[j] == 0){
					if (data.data[j] >= i)
						error_type1 += weight[j];
					else
						error_type0 += weight[j];
				}
				/*********************************************************/
				// 1, 1, 1, 1, 0, i, 1, 0 , 0, 0, 0
				// t  t  t  t  f  x  f  t   t  t  t
				// t: Classification Right; f: Classification wrong;
				//if data < classifier&&data.label == 1 is right class type = 0
				/*********************************************************/

			}
			if (error_type0 <= error_type1) {
				if (temp.error > error_type0) {
					temp.error = error_type0;
					temp.Classifier = i;
					temp.type = 0;
				}
			}
			else if (error_type1 < error_type0) {
				if (temp.error > error_type1) {
					temp.error = error_type1;
					temp.Classifier = i;
					temp.type = 1;
				}
			}

		}
	}
	else {
		for (int loop_i = 0; loop_i <data.DataSize; loop_i++) {
			int i = data.data[loop_i];
			double error_type0 = 0.0;
			double error_type1 = 0.0;
			/*********************************************************/
			// 1, 1, 1, 1, 0, i, 1, 0 , 0, 0, 0
			// f  f  f  f  t  x  t  f   f  f  f
			// t: Classification Right; f: Classification wrong;
			//if data >= classifier&&data.label == 1 is right class type = 1
			/*********************************************************/
			
			for (int j = 0; j < data.DataSize; j++) {
				if (data.label[j] == 1){
					if (data.data[j] < i)
						error_type1 += weight[j];
					else
						error_type0 += weight[j];

				}
				else if (data.label[j] == 0){
					if (data.data[j] >= i)
						error_type1 += weight[j];
					else
						error_type0 += weight[j];
				}
				/*********************************************************/
				// 1, 1, 1, 1, 0, i, 1, 0 , 0, 0, 0
				// t  t  t  t  f  x  f  t   t  t  t
				// t: Classification Right; f: Classification wrong;
				//if data < classifier&&data.label == 1 is right class type = 0
				/*********************************************************/

			}
			if (error_type0 <= error_type1) {
				if (temp.error > error_type0) {
					temp.error = error_type0;
					temp.Classifier = i;
					temp.type = 0;
				}
			}
			else if (error_type1 < error_type0) {
				if (temp.error > error_type1) {
					temp.error = error_type1;
					temp.Classifier = i;
					temp.type = 1;
				}
			}
		
		}
	}
	
	(*atomclassifier) = temp;
	
}



char ClassifierTest(int data, int *WeakCl,char *weakType, double * weakWt, int size) {
	
	//for (int i = 0; i < size; i++) {
	//	threshold += weakWt[i];
	//}
	//threshold /= 2.0;

	double test = 0.0;
	for (int j = 0; j < size; j++) {
		if (weakType[j] == 0 && data < WeakCl[j])
			test += weakWt[j];
		else if(weakType[j] == 1 && data >= WeakCl[j])
			test += weakWt[j];
		else if (weakType[j] == 0 && data >= WeakCl[j])
			test -= weakWt[j];
		else if (weakType[j] == 1 && data < WeakCl[j])
			test -= weakWt[j];
	}


	if (test < 0)
		return 0;
	else
		return 1;

}


void UpdataWeight(AdaBoostTrainData data, double beta,double *weight, AdaBoostAtomClassifier atom) {
	 
	
	if (atom.type == 0) {
		for (int i = 0; i < data.DataSize; i++) {
			if (data.label[i] == 1 && data.data[i] < atom.Classifier)
				weight[i] *= exp(-beta);
            else if (data.label[i] == 0 && data.data[i] >= atom.Classifier)
				weight[i] *= exp(-beta);
			else
				weight[i] *= exp(beta);
		}
		
	}
	else if(atom.type==1) {
		for (int i = 0; i < data.DataSize; i++) {
			if (data.label[i] == 1 && data.data[i] >= atom.Classifier)
				weight[i] *= exp(-beta);
            else if (data.label[i] == 0 && data.data[i] < atom.Classifier)
                weight[i] *= exp(-beta);
			else
				weight[i] *= exp(beta);
			
		}
	}
	//double atom_weight = log(1.0 / beta);
	//=====================================================================
	//printf("%d ,%d\n", atom.Classifier, atom.type);
	//for (int i = 0; i < data.DataSize; i++){
	//	printf("%d :%lf\n", data.data[i], weight[i]);
	//
	//}
	//printf("-----------------------------------------\n");
	//=====================================================================
	//return atom_weight;
}





char testClassifierisOkay(AdaBoostTrainData data,int *WeakCl,char *weakTp,double * weakWt,int size) {
	//double threshold = 0.0;
	
	//threshold ;
	
	for (int i = 0; i < data.DataSize; i++) {
		if (ClassifierTest(data.data[i], WeakCl, weakTp, weakWt, size) != data.label[i])
			return 0;
	}
	return 1;
}
void AdaBoostMerge(int *classifer,char *clType,double *clWeight,int * clsizeType0,int *clsizeType1){
	int totalsize = (*clsizeType0) + (*clsizeType1);
	for (int i = 0; i < totalsize; i++){
		for (int j = i+1; j < totalsize; j++){
			if (classifer[j] == classifer[i] && clType[j] == clType[i]){
				if (clType[i] == 0)
					(*clsizeType0)--;
				else if (clType[i] == 1)
					(*clsizeType1)--;
				clType[i] = 2;
				clWeight[j] += clWeight[i];
			}
		}
	}
}
AdaBoostClassifier AdaBoost(AdaBoostTrainData data,int weakclassifiersize) {
	double * weight = (double *)malloc(sizeof(double)*data.DataSize);
	int weakClassifierRealSize0 = 0;
	int weakClassifierRealSize1 = 0;
	int weakClassifierTotalSize = 0;
	if (weight == NULL) {
		printf("AdaBoostClassifier malloc weight wrong!\n");
		exit(0);
	}


	int* WeakClassifierBuffer=(int *)malloc(sizeof(int)*weakclassifiersize);
	char* WeakClassifierTypeBuffer = (char *)malloc(sizeof(char)*weakclassifiersize);
	double* WeakClassifierWeightBuffer=(double *)malloc(sizeof(double)*weakclassifiersize);

	if (WeakClassifierBuffer == NULL || WeakClassifierWeightBuffer == NULL|| WeakClassifierTypeBuffer==NULL) {
		printf("AdaBoostClassifier malloc buffer wrong!\n");
		exit(0);
	}
    InitDataWeight(data, weight);
	for (int i = 0; i < weakclassifiersize; i++) {
		AdaBoostAtomClassifier atom;
		GetBestAtomClassifier(data, &atom, weight);
		WeakClassifierBuffer[i] = atom.Classifier;
		WeakClassifierTypeBuffer[i] = atom.type;
		if (atom.type == 0)
			weakClassifierRealSize0++;
		else if (atom.type == 1)
			weakClassifierRealSize1++;
		if (atom.error == 0.0){
			WeakClassifierWeightBuffer[i] = 1;
			break;
		}
	
		double beta = 0.5*log((1.0 - atom.error) / atom.error);
		WeakClassifierWeightBuffer[i] = beta;
		UpdataWeight(data, beta, weight,atom);
		NormalizeWeight(weight, data.DataSize);
		if (testClassifierisOkay(data, WeakClassifierBuffer, WeakClassifierTypeBuffer, WeakClassifierWeightBuffer, weakClassifierRealSize0 + weakClassifierRealSize1))
			break;
		
	}



	/*Merge weak classifier*/
	weakClassifierTotalSize = weakClassifierRealSize0 + weakClassifierRealSize1;
	AdaBoostMerge(WeakClassifierBuffer, WeakClassifierTypeBuffer, WeakClassifierWeightBuffer, &weakClassifierRealSize0, &weakClassifierRealSize1);






	AdaBoostClassifier WeakClassifier;
	WeakClassifier.ClassifierType0 = (int *)malloc(sizeof(int)*weakClassifierRealSize0);
	WeakClassifier.ClassifierType1 = (int *)malloc(sizeof(int)*weakClassifierRealSize1);
	//WeakClassifier.ClassifierType = (char *)malloc(sizeof(char)*weakClassifierRealSize);
	WeakClassifier.ClassifierWeight0 = (double *)malloc(sizeof(double)*weakClassifierRealSize0);
	WeakClassifier.ClassifierWeight1 = (double *)malloc(sizeof(double)*weakClassifierRealSize1);
	int type0_i = 0;
	int type1_i = 0;
	
	for (int i = 0; i < weakClassifierTotalSize;i++) {
		
		if (WeakClassifierTypeBuffer[i] == 0){
			WeakClassifier.ClassifierType0[type0_i] = WeakClassifierBuffer[i];
			WeakClassifier.ClassifierWeight0[type0_i] = WeakClassifierWeightBuffer[i];
			type0_i++;
		}
		else if (WeakClassifierTypeBuffer[i] == 1 ){
			WeakClassifier.ClassifierType1[type1_i] = WeakClassifierBuffer[i];
			WeakClassifier.ClassifierWeight1[type1_i] = WeakClassifierWeightBuffer[i];
			type1_i++;
		}
		//WeakClassifier.ClassifierType[i] = WeakClassifierTypeBuffer[i];
		//WeakClassifier.ClassifierWeight[i] = WeakClassifierWeightBuffer[i];
	}
    WeakClassifier.ClassifierSizeType0 = type0_i;
	WeakClassifier.ClassifierSizeType1 = type1_i;

	free(WeakClassifierBuffer);
	free(WeakClassifierTypeBuffer);
	free(WeakClassifierWeightBuffer);

	return WeakClassifier;
}


void ReleaseWeakClassifier(AdaBoostClassifier WeakClassifier){
    if(WeakClassifier.ClassifierType0!=NULL&&WeakClassifier.ClassifierWeight0!=NULL){
		free(WeakClassifier.ClassifierType0);
		free(WeakClassifier.ClassifierWeight0);
    }
	if (WeakClassifier.ClassifierType1 != NULL&&WeakClassifier.ClassifierWeight1 != NULL){
		free(WeakClassifier.ClassifierType1);
		free(WeakClassifier.ClassifierWeight1);
	}
}