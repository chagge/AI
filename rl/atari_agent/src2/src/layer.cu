//layer.cu
#include "util.h" //norm rand cudamemcpyhtd checkcudaerrors
#include "layer.h"
#include <cmath>

__global__ void updateGen(value_type *d_in, value_type *grad, value_type *msq, value_type *msqGrad, value_type alpha, value_type rho, int n, int batchSize, float eps) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx>=n)
		return;
	value_type temp = grad[idx];
	msq[idx] = (rho)*msq[idx] + (1-rho)*temp*temp;
	value_type deltaX = -1.0*alpha*(temp)/(1.0*sqrt(msq[idx] + eps));
	//value_type deltaX = -1.0*((1.0*sqrt(msqGrad[idx]+eps))*temp)/(1.0*sqrt(msq[idx] + eps));
	//msqGrad[idx] = (rho)*msqGrad[idx] + (1-rho)*deltaX*deltaX;
	d_in[idx] += deltaX;
}

Layer::Layer(int inputs_, int outputs_, int kernelDim_, int stride_, value_type iRangeD_, value_type iRangeB_, int actType_) {
	inputs = inputs_;
	outputs = outputs_;
	kernelDim = kernelDim_;
	stride = stride_;
	iRangeD = iRangeD_;
	iRangeB = iRangeB_;
	actType = actType_;
}
Layer::~Layer() {
	delete[] h_data;
	delete[] h_bias;
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_bias));
	checkCudaErrors(cudaFree(d_msq));
	checkCudaErrors(cudaFree(d_grad));
	checkCudaErrors(cudaFree(d_grad_bias));
	checkCudaErrors(cudaFree(d_msq_bias));
	checkCudaErrors(cudaFree(d_msq_grad_bias));
	checkCudaErrors(cudaFree(d_msq_grad_data));
	checkCudaErrors(cudaFree(d_hist_data));
	checkCudaErrors(cudaFree(d_hist_bias));

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
}
void Layer::initBias() {
	randInit(&h_bias, &d_bias, outputs, iRangeB);
}
void Layer::initMsq() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_msq, sizeInBytes));
}
void Layer::initMsqBias() {
	int size = outputs;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_msq_bias, sizeInBytes));
}
void Layer::initGradMsq() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_msq_grad_data, sizeInBytes));
}
void Layer::initGradMsqBias() {
	int size = outputs;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_msq_grad_bias, sizeInBytes));
}
void Layer::initGrad() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_grad, sizeInBytes));
}

void Layer::initGradBias() {
	int size = outputs;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMalloc(&d_grad_bias, sizeInBytes));
}

void Layer::initHistData() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMalloc(&d_hist_data, size*sizeof(value_type)));
	checkCudaErrors(cudaMemcpyDTD(d_hist_data, d_data, size*sizeof(value_type)));
}

void Layer::initHistBias() {
	int size = outputs;
	checkCudaErrors(cudaMalloc(&d_hist_bias, size*sizeof(value_type)));
	checkCudaErrors(cudaMemcpyDTD(d_hist_bias, d_bias, size*sizeof(value_type)));
}

void Layer::init() {
	initData();
	initHistData();
	initBias();
	initHistBias();
	initMsq();
	initGrad();
	initGradBias();
	resetMsq();
	resetGrad();
	resetGradBias();
	initMsqBias();
	resetMsqBias();
	initGradMsq();
	initGradMsqBias();
	resetMsqGrad();
	resetMsqGradBias();
}
void Layer::resetMsq() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_msq, 0.0f, sizeInBytes));
}
void Layer::resetMsqBias() {
	int size = outputs;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_msq_bias, 0.0f, sizeInBytes));
}
void Layer::resetMsqGrad() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_msq_grad_data, 0.0f, sizeInBytes));
}
void Layer::resetMsqGradBias() {
	int size = outputs;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_msq_grad_bias, 0.0f, sizeInBytes));
}
void Layer::resetGrad() {
	int size = inputs*outputs*kernelDim*kernelDim;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_grad, 0.0f, sizeInBytes));
}
void Layer::resetGradBias() {
	int size = outputs;
	int sizeInBytes = size*sizeof(value_type);
	checkCudaErrors(cudaMemset(d_grad_bias, 0.0f, sizeInBytes));
}
void Layer::update(value_type alpha, value_type gamma, int batchSize, bool biasUpdate) {
	int size = inputs*outputs*kernelDim*kernelDim;
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((size-1)/threadsPerBlock.x + 1);
	updateGen<<<numBlocks, threadsPerBlock>>>(d_data, d_grad, d_msq, d_msq_grad_data, alpha, gamma, size, batchSize, 0.000001f);
	size = outputs;
	dim3 numBlocks2((size-1)/threadsPerBlock.x + 1);
	if(biasUpdate)
		updateGen<<<numBlocks2, threadsPerBlock>>>(d_bias, d_grad_bias, d_msq_bias, d_msq_grad_bias, alpha, gamma, size, batchSize, 0.000001f);
}


void Layer::copyDataDTH() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMemcpyDTH(h_data, d_data, size*sizeof(value_type)));
	size = outputs;
	checkCudaErrors(cudaMemcpyDTH(h_bias, d_bias, size*sizeof(value_type)));
}

void Layer::copyDataDTDH() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMemcpyDTD(d_hist_data, d_data, size*sizeof(value_type)));
	size = outputs;
	checkCudaErrors(cudaMemcpyDTD(d_hist_bias, d_bias, size*sizeof(value_type)));
}

void Layer::copyDataDHTD() {
	int size = inputs*outputs*kernelDim*kernelDim;
	checkCudaErrors(cudaMemcpyDTD(d_data, d_hist_data, size*sizeof(value_type)));
	size = outputs;
	checkCudaErrors(cudaMemcpyDTD(d_bias, d_hist_bias, size*sizeof(value_type)));
}