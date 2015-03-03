//layer.h
#ifndef __CUDNN_CONV_LAYER_H__
#define __CUDNN_CONV_LAYER_H__

#include "util.h"

class ConvLayer {
	public:
		value_type *d_fltr, *d_bias, *d_msq_fltr, *d_msq_bias, *d_msq_grad_fltr, *d_msq_grad_bias;
		value_type *d_grad_fltr, *d_grad_bias, *d_hist_fltr, *d_hist_bias;
		value_type *h_fltr, *h_bias;
		int inputs;
		int outputs;
		int kernelDim;
		int stride;
		value_type fltr_std;
		value_type bias_fix;
		int actType;
		Layer(int inputs_, int outputs_, int kernelDim_, int stride_, value_type fltr_std_, value_type bias_fix_, int actType_) {
			inputs = inputs_;
			outputs = outputs_;
			kernelDim = kernelDim_;
			stride = stride_;
			fltr_std = fltr_std_;
			bias_fix = bias_fix_;
			actType = actType_;
		}

		void Layer::randInit(value_type **h_dt, value_type **d_dt, int size, value_type irange, bool isFixedInit) {
			int sizeInBytes = size*sizeof(value_type);
			*h_dt = new value_type[size];
			checkCudaErrors(cudaMalloc(d_dt, sizeInBytes));
			for(int i = 0; i < size; ++i) {
				if(isFixedInit) {
					(*h_dt)[i] = value_type(irange);
				} else {
					(*h_dt)[i] = value_type(rand_normal(0, 1)*irange);
				}
			}
			checkCudaErrors(cudaMemcpyHTD(*d_dt, *h_dt, sizeInBytes));
		}

		

};

#endif