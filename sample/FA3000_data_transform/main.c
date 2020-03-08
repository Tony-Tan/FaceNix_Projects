#include "fa3000trans.h"
#include <time.h> 

int main() {
	FaForest * forest = faRead_CreateForest("D:\\Data\\FA3000\\51\\");
	faWriteData(forest);
	faFreeForest(&forest);
}