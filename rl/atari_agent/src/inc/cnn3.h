//cnn3.h
#ifndef __CNN3_H__
#define __CNN3_H__

//cnn.cpp
#include "cudnn.h"
#include <string>
#include "util.h"


class CNN {
	private:
		DIM *nnLayerDim;
		DIM *fltrLayerDim;
		DIM *stride;
		float *d_nn;
		float *d_fltr;
		float *h_fltr;
		float *d_nnerr;
		float *d_fltrerr;
		float *d_msq;
		float learnRate;
		int numSaveCntr;
		int numNNLayer;
		int numFltrLayer;
		int totalNNUnits;
		int totalFltrUnits;
		int miniBatchSize;
		int firstNNLayerUnits;
		int lastNNLayerUnits;
		int firstFltrLayerUnits;
		int lastFltrLayerUnits;
		std::string dataPath;

		cudnnHandle_t handle;
		cudnnTensorDescriptor_t *tensorDesc;
		cudnnFilterDescriptor_t *filterDesc;
		cudnnConvolutionDescriptor_t *convDesc;
	public:
		CNN(std::string, int, float, std::string);
		~CNN();
		void destroyHandles();
		void allocateLayersMemory();
		void createHandles();
		void setDescriptors();
		void initRandWts();
		void initInputLayer(float*);
		void initOutputErr(float*);
		void forwardProp();
		void backPropagate();
		void saveWeights();
		float* getQVals();
};

#endif