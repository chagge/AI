//cnn.cu
#include "cnn3.h"
#include <string>
#include <fstream>
#include <cstdlib>
#include "info.h"
#include <iostream>



__global__ void activate(float *d, int n) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx>= n)
		return;
	//d[idx] = log1pf(exp(d[idx]));
	d[idx] = 1.0/(1.0+exp(-1.0*d[idx]));
}

__global__ void calcDiffErr(float *d, float *d_, int n) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if(idx>=n)
		return;
	float t = d_[idx];
	//d[idx] = d[idx]*((exp(t)-1)/(exp(t)));
	d[idx] = d[idx]*((1-t)*t);
}

__global__ void updateWeights(float *d_f, float *d_ferr, float *d_msq, int n, float alpha) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx>=n)
		return;
	float temp = d_ferr[idx];
	d_msq[idx] = 0.9*d_msq[idx] + 0.1*temp*temp;
	if(d_msq[idx]>0.0f) {
		d_f[idx] -= (alpha*temp)/(sqrt(d_msq[idx]));
	}
}

CNN::CNN(Info info) {
	//decoding file
	std::ifstream nnConfig(info.nnConfig.c_str());
	nnConfig >> numNNLayer;
	numFltrLayer = numNNLayer - 1;
	//allocating space for dimensions of layers
	nnLayerDim = new DIM[numNNLayer];
	fltrLayerDim = new DIM[numFltrLayer];
	stride = new DIM[numFltrLayer];
	//get the first layer dimensions
	nnConfig >> nnLayerDim[0].x >> nnLayerDim[0].y >> nnLayerDim[0].z >> nnLayerDim[0].w;
	//get the filter layer and strides dimensions
	for(int i = 0; i < numFltrLayer; ++i) {
		nnConfig >> fltrLayerDim[i].x >> fltrLayerDim[i].y >> fltrLayerDim[i].z >> fltrLayerDim[i].w;
		nnConfig >> stride[i].x >> stride[i].y >> stride[i].z >> stride[i].w;
	}

	miniBatchSize = info.miniBatchSize;
	learnRate = info.lr;
	dataPath = info.dataPath;
	numSaveCntr = 0;

	createHandles();
	setDescriptors();
	allocateLayersMemory();
	initRandWts();

	firstNNLayerUnits = nnLayerDim[0].x*nnLayerDim[0].y*nnLayerDim[0].z*nnLayerDim[0].w;
	lastNNLayerUnits = nnLayerDim[numNNLayer-1].x*nnLayerDim[numNNLayer-1].y*nnLayerDim[numNNLayer-1].z*nnLayerDim[numNNLayer-1].w;
	firstFltrLayerUnits = fltrLayerDim[0].x*fltrLayerDim[0].y*fltrLayerDim[0].z*fltrLayerDim[0].w;
	lastFltrLayerUnits = fltrLayerDim[numFltrLayer-1].x*fltrLayerDim[numFltrLayer-1].y*fltrLayerDim[numFltrLayer-1].z*fltrLayerDim[numFltrLayer-1].w;
}

CNN::~CNN() {
	destroyHandles();
	delete[] nnLayerDim;
	delete[] fltrLayerDim;
	delete[] stride;
	delete[] h_fltr;
	delete[] tensorDesc;
	delete[] convDesc;
	delete[] filterDesc;
	cudaFree(d_fltr);
	cudaFree(d_nn);
	cudaFree(d_nnerr);
	cudaFree(d_fltrerr);
	cudaFree(d_msq);
}

void CNN::destroyHandles() {
	cudnnDestroy(handle);
	for(int i = 0; i < numNNLayer; ++i) {
		cudnnDestroyTensorDescriptor(tensorDesc[i]);
	}

	for(int i = 0; i < numFltrLayer; ++i) {
		cudnnDestroyFilterDescriptor(filterDesc[i]);
		cudnnDestroyConvolutionDescriptor(convDesc[i]);
	}
}

void CNN::allocateLayersMemory() {
	totalFltrUnits = 0;
	for(int i = 0; i < numFltrLayer; ++i) {
		totalFltrUnits += fltrLayerDim[i].x*fltrLayerDim[i].y*fltrLayerDim[i].z*fltrLayerDim[i].w;
	}

	totalNNUnits = 0;
	for(int i = 0; i < numNNLayer; ++i) {
		totalNNUnits += nnLayerDim[i].x*nnLayerDim[i].y*nnLayerDim[i].z*nnLayerDim[i].w;
	}

	h_fltr = new float[totalFltrUnits];
	cudaMalloc((void**)&d_nn, totalNNUnits*sizeof(float));
	cudaMalloc((void**)&d_fltr, totalFltrUnits*sizeof(float));
	cudaMalloc((void**)&d_nnerr, totalNNUnits*sizeof(float));
	cudaMalloc((void**)&d_fltrerr, totalFltrUnits*sizeof(float));
	cudaMalloc((void**)&d_msq, totalFltrUnits*sizeof(float));
}

void CNN::createHandles() {
	cudnnCreate(&handle);
	tensorDesc = new cudnnTensorDescriptor_t[numNNLayer];
	filterDesc = new cudnnFilterDescriptor_t[numFltrLayer];
	convDesc = new cudnnConvolutionDescriptor_t[numFltrLayer];
	
	for(int i = 0; i < numNNLayer; ++i) {
		cudnnCreateTensorDescriptor(&tensorDesc[i]);
	}

	for(int i = 0; i < numFltrLayer; ++i) {
		cudnnCreateFilterDescriptor(&filterDesc[i]);
		cudnnCreateConvolutionDescriptor(&convDesc[i]);
	}
}

void CNN::setDescriptors() {
	int *h_outDim = new int[NBDIMS];
	int h_padA[] = {0, 0};
	int h_upscaleA[] = {1, 1};

	//set nn layer 0 descriptor
	cudnnSetTensor4dDescriptor(tensorDesc[0], 
								CUDNN_TENSOR_NCHW,
								CUDNN_DATA_FLOAT,
								miniBatchSize,
								nnLayerDim[0].z, 
								nnLayerDim[0].x, 
								nnLayerDim[0].y);


	for(int i = 0; i < numFltrLayer; ++i) {
		
		int h_filterDimA[] = {fltrLayerDim[i].w, fltrLayerDim[i].z, fltrLayerDim[i].x, fltrLayerDim[i].y};
		int h_filterStrideA[] = {stride[i].x, stride[i].y};
		
		//set ith filter layer decriptor
		cudnnSetFilterNdDescriptor(filterDesc[i],
									CUDNN_DATA_FLOAT,
									NBDIMS,
									h_filterDimA);

		//set ith conv layer descriptor
		cudnnSetConvolutionNdDescriptor(convDesc[i],
										NBDIMS-2,
										h_padA,
										h_filterStrideA,
										h_upscaleA,
										CUDNN_CONVOLUTION);

		//get output dimensions of (i+1)th nn layer
		cudnnGetConvolutionNdForwardOutputDim(convDesc[i],
												tensorDesc[i],
												filterDesc[i],
												NBDIMS,
												h_outDim);

		//set output dimensions of (i+1)th nn layer
		nnLayerDim[i+1].w =h_outDim[0];
		nnLayerDim[i+1].z =h_outDim[1];
		nnLayerDim[i+1].x =h_outDim[2];
		nnLayerDim[i+1].y =h_outDim[3];

		//set i+1th nn layer descriptor
		cudnnSetTensor4dDescriptor(tensorDesc[i+1],CUDNN_TENSOR_NCHW,
									 CUDNN_DATA_FLOAT, 
									 miniBatchSize,
									 nnLayerDim[i+1].z,
									 nnLayerDim[i+1].x,
									 nnLayerDim[i+1].y);
	}
	delete[] h_outDim;
}

void CNN::initRandWts() {
	//in -0.01 to 0,01
	for(int i = 0; i < totalFltrUnits; ++i) {
		h_fltr[i] = (float)(rand())/(float)(RAND_MAX);
		h_fltr[i] *= 0.02f;
		h_fltr[i] -= 0.01f;
	}
	cudaMemcpyHTD(d_fltr, h_fltr, totalFltrUnits*sizeof(float));
}

void CNN::initInputLayer(float *input) {
	//caller must free input
	cudaMemcpyHTD(d_nn, input,firstNNLayerUnits*sizeof(float));
}

void CNN::initOutputErr(float *err) {
	//caller must free err
	for(int i = 0; i < lastNNLayerUnits; ++i) {
		std::cout << err[i] << " ";
	}
	int ddd;
	std::cout << std::endl;
	//std::cin >> ddd;
	cudaMemcpyHTD(d_nnerr+totalNNUnits-lastNNLayerUnits, err, lastNNLayerUnits*sizeof(float));
}

float* CNN::getQVals() {
	float *h_final = new float[lastNNLayerUnits];
	cudaMemcpyDTH(h_final, d_nn + totalNNUnits - lastNNLayerUnits, lastNNLayerUnits*sizeof(float));
	return h_final;
}


void CNN::forwardProp() {
	cudnnConvolutionFwdAlgo_t algo;
	size_t size;
	int prevNNUnits = 0;
	int prevFltrUnits = 0;
	float alpha = 1, beta = 0;
	dim3 threadsPerBlock(BLOCKSIZE);

	for(int i = 0; i < numFltrLayer; ++i) {
		int curLayerUnits = nnLayerDim[i].x*nnLayerDim[i].y*nnLayerDim[i].z*nnLayerDim[i].w;
		int nxtLayerUnits = nnLayerDim[i+1].x*nnLayerDim[i+1].y*nnLayerDim[i+1].z*nnLayerDim[i+1].w;

		//get forward algo
		cudnnGetConvolutionForwardAlgorithm(handle,
											tensorDesc[i],
										    filterDesc[i],
										    convDesc[i],
											tensorDesc[i+1], 
											CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
											0, 
											&algo);
		//get workspace size in bytes
		cudnnGetConvolutionForwardWorkspaceSize(handle,
												tensorDesc[i],
												filterDesc[i],
												convDesc[i],
												tensorDesc[i+1],
												algo,
												&size);

		void *workspace = NULL;
		if(size!=0)
			cudaMalloc((void**)&workspace, size);
		
		//forward convolution propagate
		cudnnConvolutionForward(handle,
								&alpha,
								tensorDesc[i],
								d_nn + prevNNUnits, 
								filterDesc[i], 
								d_fltr + prevFltrUnits, 
								convDesc[i],
								algo,
								workspace,
								size,
								&beta,
								tensorDesc[i+1],
								d_nn + prevNNUnits + curLayerUnits);
		
		if(size!=0)
			cudaFree(workspace);
		
		prevNNUnits += curLayerUnits;
		prevFltrUnits += fltrLayerDim[i].x*fltrLayerDim[i].y*fltrLayerDim[i].z*fltrLayerDim[i].w;
		/*RELU not working on gpu
		status = cudnnActivationForward(handle,
							   CUDNN_ACTIVATION_RELU, 
							   &alpha,
							   tensorDesc[i+1],
							   d_nn+prevNNUnits,
							   &beta, 
							   tensorDesc[i+1], 
							   d_nn+prevNNUnits);
		*/
		
		//activation function applied element wise
		dim3 numBlocks((nxtLayerUnits-1)/threadsPerBlock.x + 1);
		activate<<<numBlocks, threadsPerBlock>>>(d_nn+prevNNUnits, nxtLayerUnits);
	}
}

void CNN::backPropagate() {
	int prevNNUnits = totalNNUnits;
	int prevFltrUnits = totalFltrUnits;
	float alpha = 1;
	float beta = 0;
	//zero float memset is allowed
	cudaMemset(d_msq, 0, totalFltrUnits*sizeof(float));
	dim3 threadsPerBlock(BLOCKSIZE);

	for(int i = numNNLayer-1; i >= 1; --i) {
		
		int curNNLayerUnits = nnLayerDim[i].x*nnLayerDim[i].y*nnLayerDim[i].z*nnLayerDim[i].w;
		int curFltrLayerUnits = fltrLayerDim[i-1].x*fltrLayerDim[i-1].y*fltrLayerDim[i-1].z*fltrLayerDim[i-1].w;
		int prevNNLayerUnits = nnLayerDim[i-1].x*nnLayerDim[i-1].y*nnLayerDim[i-1].z*nnLayerDim[i-1].w;

		prevNNUnits -= curNNLayerUnits;
		prevFltrUnits -= curFltrLayerUnits;
		dim3 numBlocks((curNNLayerUnits-1)/threadsPerBlock.x + 1);

		//calculate differential error
		calcDiffErr<<<numBlocks, threadsPerBlock>>>(d_nnerr+prevNNUnits, d_nn+prevNNUnits, curNNLayerUnits);
		
		cudnnConvolutionBackwardFilter(handle,
										&alpha,
										tensorDesc[i-1],
										d_nn+prevNNUnits-prevNNLayerUnits,
										tensorDesc[i],
										d_nnerr+prevNNUnits,
										convDesc[i-1],
										&beta,
										filterDesc[i-1],
										d_fltrerr+prevFltrUnits);

		cudnnConvolutionBackwardData(handle,
										&alpha,
										filterDesc[i-1],
										d_fltr+prevFltrUnits,
										tensorDesc[i],
										d_nnerr+prevNNUnits,
										convDesc[i-1],
										&beta,
										tensorDesc[i-1],
										d_nnerr+prevNNUnits-prevNNLayerUnits);
	}
	dim3 numBlocks2((totalFltrUnits-1)/threadsPerBlock.x + 1);
	updateWeights<<<numBlocks2, threadsPerBlock>>>(d_fltr,
												 	d_fltrerr,
												 	d_msq,
												 	totalFltrUnits,
												 	learnRate);
}

void CNN::saveWeights() {
	cudaMemcpyDTH(h_fltr, d_fltr, totalFltrUnits*sizeof(float));
	std::string path = dataPath + "/WF" + toString(numSaveCntr);
	std::ofstream myF(path.c_str());
	for(int i = 0; i < totalFltrUnits; ++i)
		myF << h_fltr[i] << std::endl;
	numSaveCntr++;
	myF.close();
}
