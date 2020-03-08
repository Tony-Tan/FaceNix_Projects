#include "jcRead.h"
#include "fxBase.h"
char *exe = ".fnx";

void jcReadATree(FILE* file,JcTreeNode **root,int depth){
	if (depth < 4){
		char node_type;

		fscanf(file,"%c ", &node_type);
		switch (node_type){
		case 'A':
			(*root) = (JcTreeNode*)fxMalloc(sizeof(JcTreeNode));
			(*root)->node_type = ALIGNMENT;
			(*root)->feature_al = (JcAlignmentFeature *)fxMalloc(sizeof(JcAlignmentFeature));
			(*root)->feature_dt = NULL;
			(*root)->feature_lf = NULL;
			(*root)->left_child = NULL;
			(*root)->right_child = NULL;
			fscanf(file, "%d %d %d %d %d\n", &((*root)->feature_al->off_set[0].x),
				&((*root)->feature_al->off_set[0].y), &((*root)->feature_al->off_set[1].x),
				&((*root)->feature_al->off_set[1].y), &((*root)->feature_al->threshold));
			jcReadATree(file, &((*root)->left_child), depth + 1);
			jcReadATree(file, &((*root)->right_child), depth + 1);
			break;
		case 'D':
			(*root) = (JcTreeNode*)fxMalloc(sizeof(JcTreeNode));
			(*root)->node_type = DETECTION;
			(*root)->feature_al = NULL;
			(*root)->feature_dt =  (JcDetectionFeature *)fxMalloc(sizeof(JcDetectionFeature));
			(*root)->feature_lf = NULL;
			(*root)->left_child = NULL;
			(*root)->right_child = NULL;
			fscanf(file, "%d %d %d %d %d %d %d\n", &((*root)->feature_dt->FacialPoint[0]),
				&((*root)->feature_dt->FacialPoint[1]), &((*root)->feature_dt->off_set[0].x),
				&((*root)->feature_dt->off_set[0].y), &((*root)->feature_dt->off_set[1].x),
				&((*root)->feature_dt->off_set[1].y), &((*root)->feature_dt->threshold));
			jcReadATree(file, &((*root)->left_child), depth + 1);
			jcReadATree(file, &((*root)->right_child), depth + 1);

			break;
		case 'L':
			(*root) = (JcTreeNode*)fxMalloc(sizeof(JcTreeNode));
			(*root)->node_type = LEAF;
			(*root)->feature_al = NULL;
			(*root)->feature_dt = NULL;
			(*root)->feature_lf =(JcLeafFeature *)fxMalloc(sizeof(JcLeafFeature));
			fscanf(file, "%lf\n", &((*root)->feature_lf->class_score));
			for (int i = 0; i < LANDMARKTYPE; i++){
				fscanf(file, "%lf %lf ", &((*root)->feature_lf->off_set_array[i].x), &((*root)->feature_lf->off_set_array[i].y));
			}
			fscanf(file, "\n");
			(*root)->left_child = NULL;
			(*root)->right_child = NULL;
			break;

		}
	}
}

JcCascadeData * jcReadCascadeData(char * path){
	JcCascadeData * cas = (JcCascadeData *)fxMalloc(sizeof(JcCascadeData));

	char file_name[100];
	int landmark_num;
	double threshold;
	char c;
	for (int i = 0; i < STAGES_OF_CASCADE; i++){
		sprintf(file_name, "%s%s%s", path, "Cascade", exe);
		FILE* file = fopen(file_name, "r");
		for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
			fscanf(file ,"%c %d %lf\n", &c, &landmark_num, &threshold);
			cas->CascadeData[i][j].landmark_num = landmark_num;
			cas->CascadeData[i][j].threshold = threshold;
			jcReadATree(file, &(cas->CascadeData[i][j].root), 0);
		}
		fclose(file);
	}
	return cas;
}


void jcFreeTree(JcTreeNode * root){
	if (root->node_type == LEAF){
		free(root->feature_lf);
		free(root);
	}
	else{
		jcFreeTree(root->left_child);
		jcFreeTree(root->right_child);
		if (root->node_type == DETECTION){
			free(root->feature_dt);
			free(root);

		}
		else if (root->node_type == ALIGNMENT){
			free(root->feature_al);
			free(root);

		}
	}
	

}

void jcFreeCascadeData(JcCascadeData **casdata){
	for (int i = 0; i < STAGES_OF_CASCADE; i++){
		for (int j = 0; j < WEAK_C_R_EACH_STAGE; j++){
			jcFreeTree((*casdata)->CascadeData[i][j].root);
		}
	
	}
	fxFree(casdata);

}
