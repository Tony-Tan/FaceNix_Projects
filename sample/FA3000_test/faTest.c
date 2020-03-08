#include "faTest.h"

#include "../../data/data.inc"
void faUpdataShape(FaData* data,int lbf_num,int stage){
	
	for (int i = 0; i < LANDMARK_TYPE; i++){
		data->landmark[i].x += W[stage][i * 2][lbf_num];
		data->landmark[i].y += W[stage][i * 2 + 1][lbf_num];
	}
	
}
void faTest(FaData* data){
	
	for (int i = 0; i < STAGE; i++){
		for (int j = 0; j < TREE_NUM_EACH_STAGE; j++){
			int landmark = j % LANDMARK_TYPE;
			FaTree* tree =&( ((FaForest*) forest)->tree[i][j]);
			int lbf_num = faLBF(data, tree, landmark);
			lbf_num += j*TREE_LEAF_NUM;
			faUpdataShape(data, lbf_num, i,W);
		}
		
	}

}