#include "fxBPNerveNetwork.h"


#ifdef _OPENMP
#	include<omp.h>
#endif


static void fxBPResidual(Fx_BPTrainData *train_data, Fx_BPNerveNetwork *network){
	int num_output = network->neuron_nums_per_layer[network->layer_num - 1];
	int loop_i;
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < num_output; loop_i++){
		train_data->residual_array[loop_i] = train_data->output_array[loop_i]
			- ((network->layer[network->layer_num - 1])[loop_i]).output;
		
	}

}


static void fxZeroBPResidual(Fx_BPTrainData *train_data){
	memset(train_data->residual_array, 0, sizeof(double)*train_data->residual_num);
}

Fx_BPNerveNetwork* fxCreateBP(int * neuron_nums_per_layer,int size){
	FX_FUNCTION("fxCreateBP");
	FX_PARAM_NEGATIVE_TEST(size);
	FX_NULL_POINTER_TEST(neuron_nums_per_layer);
	__BEGIN__
	Fx_BPNerveNetwork *bp_network = (Fx_BPNerveNetwork*)fxMalloc(sizeof(Fx_BPNerveNetwork));
	bp_network->layer = (Fx_Neuron **)fxMalloc(sizeof(Fx_Neuron*)*size);
	bp_network->neuron_nums_per_layer=(int *)fxMalloc(sizeof(int)*size);
	memcpy(bp_network->neuron_nums_per_layer, neuron_nums_per_layer, sizeof(int)*size);
    bp_network->layer_num=size;

	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < size; loop_i++){
		bp_network->layer[loop_i] = (Fx_Neuron*)fxMalloc(sizeof(Fx_Neuron)*neuron_nums_per_layer[loop_i]);
	
	}
	__END__
    return bp_network;
}

#define SYNAPSE_WEIGHT_MAX_RANGE 10000
void fxCreateSynapseConnect(FILE * connect_map,Fx_BPNerveNetwork *network,Fx_ActivationFunctionType act_type){
    FX_FUNCTION("fxCreateSynapseConnect");
    __BEGIN__
    if(connect_map==NULL){
		int loop_i = 0;
#ifdef _OPENMP
#   pragma omp parallel for
#endif
        
        for(loop_i=0;loop_i<network->layer_num;loop_i++){
            for(int i=0;i<network->neuron_nums_per_layer[loop_i];i++){
                Fx_Neuron* neuron=&((network->layer[loop_i])[i]);
				neuron->actFun_type = act_type;
                neuron->net=0.0;
                neuron->output=0.0;
                neuron->sensitivity=0.0;
                if(loop_i!=0&&loop_i!=network->layer_num-1){
                    neuron->synapse_num=network->neuron_nums_per_layer[loop_i-1];
                    neuron->synapse_array=(Fx_Synapse* )fxMalloc(sizeof(Fx_Synapse)*neuron->synapse_num);
                    for(int j=0;j<neuron->synapse_num;j++){
                        neuron->synapse_array[j].pioneer=&((network->layer[loop_i-1])[j]);

                        int weight=fxRandom(SYNAPSE_WEIGHT_MAX_RANGE, 0);
                        neuron->synapse_array[j].w=(((double)weight)*2.0/SYNAPSE_WEIGHT_MAX_RANGE);
						neuron->synapse_array[j].delta_w = 0.0;
                    }
                    neuron->neuron_type=FX_NEURON_HIDDEN;
                }else if(loop_i==0){
                    neuron->synapse_num=0;
                    neuron->neuron_type=FX_NEURON_INPUT;
                }else if(loop_i==network->layer_num-1){
                    neuron->synapse_num=network->neuron_nums_per_layer[loop_i-1];
                    neuron->synapse_array=(Fx_Synapse*)fxMalloc(sizeof(Fx_Synapse)*neuron->synapse_num);
                    for(int j=0;j<neuron->synapse_num;j++){
                        neuron->synapse_array[j].pioneer=&((network->layer[loop_i-1])[j]);
                        int weight=fxRandom(SYNAPSE_WEIGHT_MAX_RANGE, 0);
                        neuron->synapse_array[j].w=((double)weight)*2.0/SYNAPSE_WEIGHT_MAX_RANGE;
						neuron->synapse_array[j].delta_w = 0.0;
                    }
                    neuron->neuron_type=FX_NEURON_OUTPUT;
                }
            
            }
        }
    }
    __END__
}

void fxFreeBP(Fx_BPNerveNetwork** network){
    int loop_i=0;
#ifdef _OPENMP
#   pragma omp parallel for
#endif
    for(loop_i=0;loop_i<(*network)->layer_num;loop_i++){
        int neuron_num_this_layer=(*network)->neuron_nums_per_layer[loop_i];
        if(loop_i!=0){
            for(int i=0;i<neuron_num_this_layer;i++){
                Fx_Neuron * neuron=&(((*network)->layer[loop_i])[i]);
                fxFree((void *)&(neuron->synapse_array));
            }
        }
        fxFree((void *)&((*network)->layer[loop_i]));
    }
    fxFree((void *)network);
}


double fxActivationFun(double net,Fx_ActivationFunctionType type){
    FX_FUNCTION("fxActivationFun");
    __BEGIN__
    double act_value;
    switch (type) {
        case FX_SIFMOID:{
            act_value=1.0/(1+exp(-1.0*net));
            break;
        }
		case FX_TANH:{
			double e_z = exp(net);
			double m_e_z = exp(-net);
			act_value = (e_z - m_e_z) / (e_z + m_e_z);
			break;
		}
        default:
            break;
    }
    
    
    __END__

    return act_value;

}

double fxActivationFunDer(double net, Fx_ActivationFunctionType type){
	FX_FUNCTION("fxActivationFunDer");
	__BEGIN__
	double act_value_der=0.0;
	double act_value = fxActivationFun(net, type);
	switch (type) {
	case FX_SIFMOID:{
		act_value_der = act_value*(1.0 - act_value);
		break;
	}
	case FX_TANH:{
		act_value_der = (1.0 - act_value*act_value);
		break;
	}
	default:
		break;
	}


	__END__
		return act_value_der;

}

void fxFPropagation(Fx_BPTrainData* train_data, Fx_BPNerveNetwork * network){
	FX_FUNCTION("fxFPropagation");
    FX_NULL_POINTER_TEST(train_data);
    FX_NULL_POINTER_TEST(network);
    __BEGIN__
    int layer_num=network->layer_num;
	
    for(int i=0;i<layer_num;i++){
		int neuron_num = network->neuron_nums_per_layer[i];
		if (i == 0){
			
			for (int j = 0; j < neuron_num; j++){
				Fx_Neuron *neuron = &((network->layer[i])[j]);
				neuron->output = train_data->input_array[j];
			}
		}
		else{
			int loop_i = 0;
			
#ifdef _OPENMP
#   pragma omp parallel for
#endif
			for (loop_i = 0; loop_i < neuron_num; loop_i++){
				Fx_Neuron *neuron = &((network->layer[i])[loop_i]);
				int synapse_num = neuron->synapse_num;
				double net_value = 0.0;
				for (int j = 0; j < synapse_num; j++){
					net_value += (neuron->synapse_array[j].w*
						(neuron->synapse_array[j].pioneer)->output);
				}
				neuron->net = net_value;
				neuron->output = fxActivationFun(net_value, neuron->actFun_type);
			}
		}
    }
	
	fxBPResidual(train_data, network);
	
    __END__
}

static double fxBPSensitivity(Fx_BPNerveNetwork *network, Fx_BPTrainData *train_data, int num_in_layer, int layer_num){
	Fx_Neuron *neuron = &((network->layer[layer_num])[num_in_layer]);

	if (neuron->neuron_type == FX_NEURON_OUTPUT){
		neuron->sensitivity = fxActivationFunDer(neuron->net, neuron->actFun_type)*
									train_data->residual_array[num_in_layer];
		if (fxActivationFunDer(neuron->net, neuron->actFun_type)< 0){
			
			printf("stop");
		
		}
		
	}
	else if (neuron->neuron_type == FX_NEURON_HIDDEN){
		double act_der = fxActivationFunDer(neuron->net, neuron->actFun_type);
		double sum_next_layer = 0.0;
		int next_layer_num = layer_num + 1;
		int next_layer_neuron_num = network->neuron_nums_per_layer[next_layer_num];
		int loop_i = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction (+ : sum_next_layer)
#endif 
		for (loop_i = 0; loop_i < next_layer_neuron_num; loop_i++){
			sum_next_layer+=(network->layer[next_layer_num])[loop_i].synapse_array[num_in_layer].w*
							(network->layer[next_layer_num])[loop_i].sensitivity;
		
		}
		neuron->sensitivity = fxActivationFunDer(neuron->net, neuron->actFun_type)*sum_next_layer;
	
	}
	

}

void fxBPNetWorkSensitivity(Fx_BPTrainData *train_data, Fx_BPNerveNetwork *network){
	int layer_num = network->layer_num;
	int* neuron_num_per_layer = network->neuron_nums_per_layer;

	for (int j = layer_num-1; j >0; j--){
		for (int i = 0; i < neuron_num_per_layer[j]; i++){
			fxBPSensitivity(network, train_data, i, j);
		}
	
	}
}




void fxBPropagation(Fx_BPTrainData* train_data, Fx_BPNerveNetwork * network, double step){
	FX_FUNCTION("fxBPropagation");
	FX_NULL_POINTER_TEST(train_data);
	FX_NULL_POINTER_TEST(network);
	__BEGIN__
	int layer_num = network->layer_num;

	for (int i = layer_num-1; i>0; i--){
		int neuron_num_this_layer = network->neuron_nums_per_layer[i];
		int loop_i = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (loop_i = 0; loop_i < neuron_num_this_layer; loop_i++){
			Fx_Neuron* neuron = &((network->layer[i])[loop_i]);
			double sensitivity = neuron->sensitivity;
			for (int j = 0; j < neuron->synapse_num; j++){
				Fx_Synapse *synapse = &(neuron->synapse_array[j]);
				synapse->delta_w = step*sensitivity*synapse->pioneer->output;
			}
		}
	}
	__END__

}
void fxUpdateWeigth(Fx_BPNerveNetwork * network){
	int layer_num = network->layer_num;
	int* neuron_num_per_layer = network->neuron_nums_per_layer;
	int loop_i = 1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (loop_i = 1; loop_i < layer_num; loop_i++){
		for (int j = 0; j < neuron_num_per_layer[loop_i]; j++){
			Fx_Neuron * neuron = &((network->layer[loop_i])[j]);
			int synapse_num = neuron->synapse_num;
			for (int i = 0; i < synapse_num; i++){
				(neuron->synapse_array[i].w) += (neuron->synapse_array[i].delta_w);
			}
		}
	}


}



double fxTestBP(Fx_BPNerveNetwork *network,Fx_BPTrainDataSet *testdata){
	//showData(testdata);
	Fx_BPTrainData* testdata_set = testdata->traindata_array;
	int testdata_size = testdata->size;
	double different = 0.0;
	for (int i = 0; i < testdata_size; i++){
		fxFPropagation(&(testdata_set[i]),network);
		for (int j = 0; j < testdata_set[i].residual_num; j++){
			different +=  ((testdata_set[i].residual_array)[j]) *((testdata_set[i].residual_array)[j]);
		}
	}
	return different / 2.0;

}

static void fxUpDataTrainDataStatus(Fx_BPTrainDataSet *traindata_set){
	int data_size = traindata_set->size;
	int loop_i = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for (loop_i = 0; loop_i < data_size; loop_i++){
		traindata_set->traindata_array[loop_i].states = 1;
	}
}


void fxTrainBP(Fx_BPTrainDataSet *traindata_set, Fx_BPNerveNetwork *network,
	Fx_BPTrainDataSet *testdata, double step,double theta,Fx_BPTrainType type){
	FX_FUNCTION("fx_TrainBPNerveNetwork");

	FX_NULL_POINTER_TEST(traindata_set);
	FX_NULL_POINTER_TEST(network);
	__BEGIN__
	showData(traindata_set);
	showNet(network);
	int traindata_size = traindata_set->size;
	Fx_BPTrainData *traindata = traindata_set->traindata_array;

	switch (type)
	{
		case FX_BP_BATCH:{

		
							 break;
		}
		case FX_BP_STOCHASTIC:{
			int epoch = 0;
			double diff = fxTestBP(network, testdata);
			while (diff >= theta){
				if (!(epoch % 100)){
					showNet(network);
					printf("diff:%lf  train epoch:%d\n", diff, epoch);
				}
				epoch++;
				fxUpDataTrainDataStatus(traindata_set);
				
				for (int i = 0; i < traindata_size;){
					if (i < traindata_size *2/ 3){
						int random_data_num = fxRandom(traindata_size, FX_RANDOM_MEAN_HALF_N);
						
						if (traindata[random_data_num].states){
							traindata[random_data_num].states = 0;
							fxFPropagation(&(traindata[random_data_num]), network);
							fxBPNetWorkSensitivity(&(traindata[random_data_num]), network);
							fxBPropagation(&(traindata[random_data_num]), network, step);
							fxUpdateWeigth(network);
							i++;
							

						}
					}else{
						for (int j = 0; j < traindata_size; j++){
							if (traindata[j].states){
								traindata[j].states = 0;
								fxFPropagation(&(traindata[j]), network);
								fxBPNetWorkSensitivity(&(traindata[j]), network);
								fxBPropagation(&(traindata[j]), network, step);
								fxUpdateWeigth(network);
							}
						}
						break;

					}
				}

				diff = fxTestBP(network, testdata);
				}
			break;
		}
		default:
			break;
	}
	__END__

}

void fxSaveBPNet(char *file_path, Fx_BPNerveNetwork *network){
	FX_FUNCTION("FxSaveBP");
	FX_NULL_POINTER_TEST(file_path);
	FX_NULL_POINTER_TEST(network);
	__BEGIN__
	
	
	{
		char pro_filename[256];
		sprintf(pro_filename, "%s%s",file_path, "pro_file.bp"); 
		FILE *file = fopen(pro_filename, "w+");
		fprintf(file,"layer_num:%d\n", network->layer_num);
		fprintf(file, "neuron_nums_per_layer:");
		for (int i = 0; i < network->layer_num; i++){
			fprintf(file, "%d ",network->neuron_nums_per_layer[i]);
		
		}
		fprintf(file, "\n");
		fclose(file);
	}



	int layer_num = network->layer_num;

	for (int loop_i=0; loop_i < layer_num; loop_i++){
		char file_name[256];
		int neuron_nums_per_alyer = network->neuron_nums_per_layer[loop_i];
		sprintf(file_name, "%s%d%s", file_path, loop_i, ".bp");
		FILE *file = fopen(file_name, "w+");
		for (int j = 0; j < neuron_nums_per_alyer; j++){
			Fx_Neuron * neuron = &((network->layer[loop_i])[j]);
			fprintf(file, "Num:%d\nActFun_type:%d\nNeuron_type:%d\nSynapse_num:%d\n", 
							j,neuron->actFun_type,neuron->neuron_type,neuron->synapse_num);
			if (neuron->neuron_type != FX_NEURON_INPUT){
				for (int i = 0; i < neuron->synapse_num; i++){
					fprintf(file, "%lf ", neuron->synapse_array[i].w);

				}
				
			}
			fprintf(file, "\n");
		}
		fclose(file);
	}
	__END__

}



Fx_BPNerveNetwork * FxReadBP(char *file_path){
	FX_FUNCTION("FxReadBP");
	FX_NULL_POINTER_TEST(file_path);
	__BEGIN__
	
	Fx_BPNerveNetwork *network=(Fx_BPNerveNetwork* )fxMalloc(sizeof(Fx_BPNerveNetwork));
	{
		char pro_filename[256];
		sprintf(pro_filename,"%s%s", file_path, "pro_file.bp");
		FILE *file = fopen(pro_filename, "r");
		fscanf(file, "layer_num:%d\n", &network->layer_num);
		network->neuron_nums_per_layer = (int *)fxMalloc(sizeof(int)*network->layer_num);

		fscanf(file, "neuron_nums_per_layer:");
		for (int i = 0; i < network->layer_num; i++){
			fscanf(file, "%d ", &(network->neuron_nums_per_layer[i]));
			//printf("%d ", (network->neuron_nums_per_layer[i]));
		}
		fscanf(file, "\n");
		fclose(file);
	}
	


	int layer_num = network->layer_num;
	network->layer = (Fx_Neuron**)fxMalloc(sizeof(Fx_Neuron*)*layer_num);
	for (int loop_i = 0; loop_i < layer_num; loop_i++){
		char file_name[256];
		int neuron_nums_per_alyer = network->neuron_nums_per_layer[loop_i];
		network->layer[loop_i] = (Fx_Neuron *)fxMalloc(sizeof(Fx_Neuron)*neuron_nums_per_alyer);
		sprintf(file_name, "%s%d%s", file_path, loop_i, ".bp");
		FILE *file = fopen(file_name, "r");
		for (int i = 0; i < neuron_nums_per_alyer;i++ ){
			int j;
			Fx_Neuron * neuron = &((network->layer[loop_i])[i]);
			fscanf(file, "Num:%d\nActFun_type:%d\nNeuron_type:%d\nSynapse_num:%d\n",
				&j, &(neuron->actFun_type), &(neuron->neuron_type), &(neuron->synapse_num));
			printf("\nNum:%d\nActFun_type:%d\nNeuron_type:%d\nSynapse_num:%d\n",
				j, (neuron->actFun_type), (neuron->neuron_type), (neuron->synapse_num));
			if (neuron->neuron_type != FX_NEURON_INPUT){
				neuron->synapse_array = (Fx_Synapse*)fxMalloc(sizeof(Fx_Synapse)*neuron->synapse_num);

				for (int i = 0; i < neuron->synapse_num; i++){
					fscanf(file, "%lf ", &(neuron->synapse_array[i].w));
					printf("%lf ", (neuron->synapse_array[i].w));
					neuron->synapse_array[i].pioneer = &(network->layer[loop_i-1][i]);
					neuron->synapse_array[i].delta_w = 0.0;
				}
				
			}

			fscanf(file, "\n");
			neuron->net = 0.0;
			neuron->output = 0.0;
			neuron->sensitivity = 0.0;
		}
		fclose(file);
	}
	__END__

	return network;
}



void fxBP(Fx_BPTestData* test_data, Fx_BPNerveNetwork * network){
	FX_FUNCTION("fxBP");
	FX_NULL_POINTER_TEST(test_data);
	FX_NULL_POINTER_TEST(network);
	__BEGIN__
	int layer_num = network->layer_num;

	for (int i = 0; i<layer_num; i++){
		int neuron_num = network->neuron_nums_per_layer[i];
		if (i == 0){

			for (int j = 0; j < neuron_num; j++){
				Fx_Neuron *neuron = &((network->layer[i])[j]);
				neuron->output = test_data->input_array[j];
			}
		}
		else{
			int loop_i = 0;

#ifdef _OPENMP
#   pragma omp parallel for
#endif
			for (loop_i = 0; loop_i < neuron_num; loop_i++){
				Fx_Neuron *neuron = &((network->layer[i])[loop_i]);
				int synapse_num = neuron->synapse_num;
				double net_value = 0.0;
				for (int j = 0; j < synapse_num; j++){
					net_value += (neuron->synapse_array[j].w*
						(neuron->synapse_array[j].pioneer)->output);
				}
				neuron->net = net_value;
				neuron->output = fxActivationFun(net_value, neuron->actFun_type);
			}
			//showNet(network);
		}
	}
	
	for (int i = 0; i < test_data->output_num; i++){
		test_data->output_array[i] = network->layer[layer_num - 1][i].output;
	}
	__END__
}

void fxSaveBPResult(char * result_file,Fx_BPTestDataSet* test_dataset ){
	FILE* file = fopen(result_file, "w+");
	int set_size = test_dataset->size;
	int out_put_size = test_dataset->testdata_array[0].output_num;
	int in_put_size = test_dataset->testdata_array[0].input_num;
	for (int i = 0; i < set_size; i++){
		for (int j = 0; j < in_put_size; j++){
			fprintf(file, "%lf ", test_dataset->testdata_array[i].input_array[j]);
		}
		fprintf(file, "    =>   ");
		for (int j = 0; j < out_put_size; j++){
			fprintf(file, "%lf ", test_dataset->testdata_array[i].output_array[j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}





/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
void showData(Fx_BPTrainDataSet *dataset){
	for (int j = 0; j < dataset->size; j++){
		printf("input :");
		for (int i = 0; i <dataset->traindata_array[j].input_num; i++){
			printf("%lf ", dataset->traindata_array[j].input_array[i]);
		}
		printf("\noutput :");
		for (int i = 0; i <dataset->traindata_array[j].output_num; i++){
			printf("%lf ", dataset->traindata_array[j].output_array[i]);
		}
		printf("\n");
	}
}

void showNet(Fx_BPNerveNetwork *network){
	int layer = network->layer_num;
	int *neurons_per_layer = network->neuron_nums_per_layer;
	printf("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < layer; i++){
		
		printf("layer: %d\n", i);
		for (int j = 0; j < neurons_per_layer[i]; j++){
			printf("_____________________________________________________\n");
			if (network->layer[i][j].neuron_type != FX_NEURON_INPUT){
				int sy_num = network->layer[i][j].synapse_num;
				printf("	neuron:%d type:%d\n    ", j, network->layer[i][j].neuron_type);
				for (int s = 0; s < sy_num; s++){
					printf("%lf      ", network->layer[i][j].synapse_array[s]);
				}
				printf("\n");
			}
			printf("	net:%lf\n", network->layer[i][j].net);
			printf("	output:%lf\n", network->layer[i][j].output);
			printf("	sensitivity:%lf\n", network->layer[i][j].sensitivity);
			printf("_____________________________________________________\n");
		
		}
	}

}
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/