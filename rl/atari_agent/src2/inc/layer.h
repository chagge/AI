//layer.h
#ifndef __LAYER_H__
#define __LAYER_H__

#include "util.h"

class Layer {
	public:
		value_type *h_data, *d_data, *d_hist_data, *d_hist_bias;
		value_type *h_bias, *d_bias;
		value_type *d_msq, *d_grad, *d_bias_msq, *d_grad_bias, *d_msq_grad_data, *d_msq_grad_bias;
		int inputs;
		int outputs;
		int kernelDim;
		int stride;
		value_type iRangeD, iRangeB;
		Layer(int, int, int, int, value_type, value_type);
		~Layer();
		void randInit(value_type**, value_type**, int, value_type);
		void initData();
		void initBias();
		void initMsq();
		void initGrad();
		void init();
		void resetMsq();
		void resetGrad();
		void update(value_type, value_type, int);
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
};

#endif