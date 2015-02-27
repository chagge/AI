//layer.h
#ifndef __CUDNN_CONV_LAYER_H__
#define __CUDNN_CONV_LAYER_H__

#include "util.h"

class Layer {
	public:
		value_type *h_data, *d_data, *d_hist_data, *d_hist_bias;
		value_type *h_bias, *d_bias;
		value_type *d_msq, *d_grad, *d_msq_bias, *d_grad_bias, *d_msq_grad_data, *d_msq_grad_bias;
		int inputs;
		int outputs;
		int kernelDim;
		int stride;
		value_type iRangeD, iRangeB;
		int actType;				// 0 == linear , 1 == relu, 2 == leaky relu
		int lType;					// 0 == ip, 1 == conv
		Layer(int, int, int, int, value_type, value_type, int, int);
		~Layer();
		void randInit(value_type**, value_type**, int, value_type, bool);
		void initData();
		void initBias();
		void initMsq();
		void initGrad();
		void init();
		void resetMsq();
		void resetGrad();
		void update(value_type, value_type, int, bool);
		void copyDataDTH();
		void copyDataDTDH();
		void copyDataDHTD();
		void initHistData();
		void initHistBias();
		void initGradBias();
		void resetGradBias();
		void initMsqBias();
		void resetMsqBias();
		void initGradMsq();
		void initGradMsqBias();
		void resetMsqGrad();
		void resetMsqGradBias();
		void rescaleWeights();
		void rescaleBias();
};

#endif