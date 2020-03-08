#ifndef BPNERVENETWORK_H
#define BPNERVENETWORK_H
#include "fxBase.h"
#include "fxError.h"
#include "fxTypes.h"
#include <math.h>
typedef struct Fx_Neuron_ Fx_Neuron;
typedef struct Fx_Synapse_ Fx_Synapse;
typedef struct Fx_BPNerveNetwork_ Fx_BPNerveNetwork;
typedef struct Fx_BPTrainData_ Fx_BPTrainData;
typedef struct Fx_BPTrainDataSet_ Fx_BPTrainDataSet;
typedef struct Fx_BPTestData_ Fx_BPTestData;
typedef struct Fx_BPTestDataSet_ Fx_BPTestDataSet;

typedef enum Fx_NeuronType_ Fx_NeuronType;
typedef enum Fx_ActivationFunctionType_ Fx_ActivationFunctionType;
typedef enum Fx_BPTrainType_ Fx_BPTrainType;


enum Fx_BPTrainType_{
	FX_BP_STOCHASTIC=0,
	FX_BP_BATCH=1,
	FX_BP_ON_LINE=2
};
enum Fx_NeuronType_{
	FX_NEURON_INPUT  = 0,
	FX_NEURON_HIDDEN = 1,
	FX_NEURON_OUTPUT = 2
};
enum Fx_ActivationFunctionType_{
	FX_SIFMOID=0,
	FX_TANH=1
};
struct Fx_Neuron_{
	Fx_ActivationFunctionType actFun_type;
	Fx_NeuronType neuron_type;	//神经元类型
	Fx_Synapse * synapse_array;	//突触数组
	int synapse_num;			//突触数量
	double sensitivity;			// 敏感度
	double net;	
	double output;				//input类型时output值表示输入值	

};
struct Fx_Synapse_{
	double w;
	double delta_w;
	Fx_Neuron* pioneer;
};

struct Fx_BPNerveNetwork_{
	int layer_num;
	Fx_Neuron ** layer;
	int * neuron_nums_per_layer;

};


struct Fx_BPTrainData_{
	int states;
	double *input_array;
	int input_num;
	double *output_array;
	int output_num;
	double *residual_array;
	int residual_num;
};
struct Fx_BPTestData_{
	double *input_array;
	int input_num;
	double *output_array;
	int output_num;
};
struct Fx_BPTrainDataSet_{
	int size;
	Fx_BPTrainData* traindata_array;

};
struct Fx_BPTestDataSet_{
	int size;
	Fx_BPTestData* testdata_array;

};


void showNet(Fx_BPNerveNetwork *network);
void showData(Fx_BPTrainDataSet *dataset);




void fxFPropagation(Fx_BPTrainData* train_data, Fx_BPNerveNetwork * network);
Fx_BPNerveNetwork* fxCreateBP(int * neuron_nums_per_layer, int size);
void fxCreateSynapseConnect(FILE * connect_map, Fx_BPNerveNetwork *network,Fx_ActivationFunctionType act_type);
void fxFreeBP(Fx_BPNerveNetwork** network);
void fxTrainBP(Fx_BPTrainDataSet *traindata_set, Fx_BPNerveNetwork *network,
	Fx_BPTrainDataSet *testdata, double step, double theta, Fx_BPTrainType type);
void fxSaveBPNet(char *file_path, Fx_BPNerveNetwork *network);
Fx_BPNerveNetwork * FxReadBP(char *file_path);
void fxSaveBPResult(char * result_file, Fx_BPTestDataSet* test_dataset);
void fxBP(Fx_BPTestData* test_data, Fx_BPNerveNetwork * network);
#endif