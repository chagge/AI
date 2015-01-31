//netowrk.h
#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "cudnn.h"
#include "util.h"
#include "layer.h"

class Network {
	private:
		cudnnDataType_t dataType;
	    cudnnTensorFormat_t tensorFormat;
	    cudnnHandle_t cudnnHandle;
	    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc, dataGradTensorDesc, diffTensorDesc;
	    cudnnFilterDescriptor_t filterDesc, filterGradDesc;
	    cudnnConvolutionDescriptor_t convDesc;
	    void createHandles();
	    void destroyHandles();
	public:
		Network();
		~Network();
		void resize(int, value_type**);
	    void addBias(const cudnnTensorDescriptor_t&, const Layer&, int, int, int, int, value_type*);
	    void convoluteForward(const Layer&, int&, int&, int&, int&, value_type*, value_type**);
        void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData);
        void convoluteBacwardData(const Layer&, int&, int&, int&, int&, value_type*, int&, int&, int&, int&, value_type**);
        void convoluteBacwardFilter(const Layer&,int&, int&, int&, int&,value_type*,int&, int&, int&, int&, value_type*, value_type**);
        void activationBackward(int&, int&, int&, int&,value_type*,value_type*, value_type*, value_type**);
        void activationForwardLeakyRELU(int, int, int, int, value_type*, value_type**, value_type);
		void activationBackwardLeakyRELU(int&, int&, int&, int&,value_type*,value_type*, value_type*, value_type**, value_type);
};

#endif