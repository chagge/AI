//layer.cu
#include "util.h" //norm rand cudamemcpyhtd checkcudaerrors
#include "layer.h"
#include <cmath>

__global__ void updateFilter(value_type *d_in, value_type *grad, value_type *msq, value_type alpha, value_type gamma, int n, int batchSize) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx>=n)
		return;
	value_type temp = grad[idx];
	msq[idx] = (1-gamma)*msq[idx] + gamma*temp*temp;
	if(msq[idx] > 0.0f) {
		d_in[idx] -= (alpha/(1.0*batchSize))*(temp/sqrt(msq[idx]));
	}
}

Layer::Layer(int inputs_, int outputs_, int kernelDim_, int stride_, value_type iRangeD_, value_type iRangeB_) {
	inputs = inputs_;
	outputs = outputs_;
	kernelDim = kernelDim_;
	stride = stride_;
	iRangeD = iRangeD_;
	iRangeB = iRangeB_;
}
Layer::~Layer() {
	delete[] h_data;
	delete[] h_bias;
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_bias));
	checkCudaErrors(cudaFree(d_msq));
	checkCudaErrors(cudaFree(d_grad));
}
void Layer::randInit(value_type **h_dt, value_type **d_dt, int size, value_type irange) {
	int sizeInBytes = size*sizeof(value_type);
	*h_dt = new value_type[size];
	checkCudaErrors(cudaMalloc(d_dt, sizeInBytes));
	for(int i = 0; i < size; ++i) {
		(*h_dt)[i] = value_type(rand_normal(0, irange));
	}
	checkCudaErrors(cudaMemcpyHTD(*d_dt, *h_dt, sizeInBytes));
}
void Layer::initData() {
	randInit(&h_data, &d_data, inputs*outputs*kernelDim*kernelDim, iRangeD);
	#ifdef TEST
		std::cout << "Layer initData: done!" << std::endl;
	#endif
}
void Layer::initBias() {
	randInit(&h_bias, &d_bias, outputs, iRangeB);
	#ifdef TEST
		std::cout << "Layer initBias: done!" << std::endl;
	#endif
}
void Layer::initMsq() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_msq, sizeInBytes));
}
void Layer::initGrad() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_grad, sizeInBytes));
}
void Layer::initHistData() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMalloc(&d_hist_data, size*sizeof(value_type)));
	checkCudaErrors(cudaMemcpyDTD(d_hist_data, d_data, size*sizeof(value_type)));
}

void Layer::init() {
	initData();
	initHistData();
	initBias();
	initMsq();
	initGrad();
	resetMsq();
	resetGrad();
}
void Layer::resetMsq() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_msq, 0.0f, sizeInBytes));
}
void Layer::resetGrad() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_grad, 0.0f, sizeInBytes));
}
void Layer::update(value_type alpha, value_type gamma, int batchSize) {
	int size = inputs*outputs*kernelDim*kernelDim;
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((size-1)/threadsPerBlock.x + 1);
	updateFilter<<<numBlocks, threadsPerBlock>>>(d_data, d_grad, d_msq, alpha, gamma, size, batchSize);
}

void Layer::copyDataDTH() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMemcpyDTH(h_data, d_data, size*sizeof(value_type)));
}

void Layer::copyDataDTDH() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMemcpyDTD(d_hist_data, d_data, size*sizeof(value_type)));
}

void Layer::copyDataDHTD() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMemcpyDTD(d_data, d_hist_data, size*sizeof(value_type)));
}