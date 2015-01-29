#include "cudnn.h"
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

#define EXIT_WAIVED 0

#define cudaMemcpyHTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost)
#define cudaMemcpyDTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice)
#define value_type float
#define BLOCKSIZE 512

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
}

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

class Layer {
	public:
		value_type *h_data, *d_data;
		value_type *h_bias, *d_bias;
		value_type *d_msq, *d_grad;
		int inputs;
		int outputs;
		int kernelDim;
		int stride;
		value_type iRangeD, iRangeB;
		Layer(int inputs_, int outputs_, int kernelDim_, int stride_, value_type iRangeD_, value_type iRangeB_) {
			inputs = inputs_;
			outputs = outputs_;
			kernelDim = kernelDim_;
			stride = stride_;
			iRangeD = iRangeD_;
			iRangeB = iRangeB_;
		}
		~Layer() {
			delete[] h_data;
			delete[] h_bias;
			checkCudaErrors(cudaFree(d_data));
			checkCudaErrors(cudaFree(d_bias));
			checkCudaErrors(cudaFree(d_msq));
			checkCudaErrors(cudaFree(d_grad));
		}
		void randInit(value_type **h_dt, value_type **d_dt, int size, value_type irange) {
			int sizeInBytes = size*sizeof(value_type);
			*h_dt = new value_type[size];
			checkCudaErrors(cudaMalloc(d_dt, sizeInBytes));
			for(int i = 0; i < size; ++i) {
				(*h_dt)[i] = irange*((value_type)(rand()))/((value_type)RAND_MAX) - irange/2.0;
			}
			checkCudaErrors(cudaMemcpyHTD(*d_dt, *h_dt, sizeInBytes));
		}
		void initData() {
			randInit(&h_data, &d_data, inputs*outputs*kernelDim*kernelDim, iRangeD);
			#ifdef TEST
				std::cout << "Layer initData: done!" << std::endl;
			#endif
		}
		void initBias() {
			randInit(&h_bias, &d_bias, outputs, iRangeB);
			#ifdef TEST
				std::cout << "Layer initBias: done!" << std::endl;
			#endif
		}
		void initMsq() {
			int size = inputs*outputs*kernelDim*kernelDim;
			int sizeInBytes = size*sizeof(value_type);
			checkCudaErrors(cudaMalloc(&d_msq, sizeInBytes));
		}
		void initGrad() {
			int size = inputs*outputs*kernelDim*kernelDim;
			int sizeInBytes = size*sizeof(value_type);
			checkCudaErrors(cudaMalloc(&d_grad, sizeInBytes));
		}
		void init() {
			initData();
			initBias();
			initMsq();
			initGrad();
			resetMsq();
			resetGrad();
		}
		void resetMsq() {
			int size = inputs*outputs*kernelDim*kernelDim;
			int sizeInBytes = size*sizeof(value_type);
			checkCudaErrors(cudaMemset(d_msq, 0.0f, sizeInBytes));
		}
		void resetGrad() {
			int size = inputs*outputs*kernelDim*kernelDim;
			int sizeInBytes = size*sizeof(value_type);
			checkCudaErrors(cudaMemset(d_grad, 0.0f, sizeInBytes));
		}
		void update(value_type alpha, value_type gamma, int batchSize) {
			int size = inputs*outputs*kernelDim*kernelDim;
			dim3 threadsPerBlock(BLOCKSIZE);
			dim3 numBlocks((size-1)/threadsPerBlock.x + 1);
			updateFilter<<<numBlocks, threadsPerBlock>>>(d_data, d_grad, d_msq, alpha, gamma, size, batchSize);
		}
};

void printDeviceVector(int size, value_type *d_vec) {
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, d_vec, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
    delete[] vec;
}

void printHostVector(int size, value_type *h_vec) {
    for (int i = 0; i < size; i++)
    {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

class Network {
	private:
		cudnnDataType_t dataType;
	    cudnnTensorFormat_t tensorFormat;
	    cudnnHandle_t cudnnHandle;
	    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc, dataGradTensorDesc, diffTensorDesc;
	    cudnnFilterDescriptor_t filterDesc, filterGradDesc;
	    cudnnConvolutionDescriptor_t convDesc;
	    
	    void createHandles() {
			checkCUDNN(cudnnCreate(&cudnnHandle));
			checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
			checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
			checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
			checkCUDNN(cudnnCreateTensorDescriptor(&dataGradTensorDesc));
			checkCUDNN(cudnnCreateTensorDescriptor(&diffTensorDesc));
			checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
			checkCUDNN(cudnnCreateFilterDescriptor(&filterGradDesc));
			checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	    }

	    void destroyHandles() {
	    	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	        checkCUDNN(cudnnDestroyFilterDescriptor(filterGradDesc));
	        checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	        checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
	        checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	        checkCUDNN(cudnnDestroyTensorDescriptor(dataGradTensorDesc));
	        checkCUDNN(cudnnDestroyTensorDescriptor(diffTensorDesc));
	        checkCUDNN(cudnnDestroy(cudnnHandle));
	    }

	public:
		Network() {
			dataType = CUDNN_DATA_FLOAT;
			tensorFormat = CUDNN_TENSOR_NCHW;
			createHandles();
		}

		~Network() {
			destroyHandles();
		}
		void resize(int size, value_type **data) {
	        if(*data != NULL)
	        {
	            checkCudaErrors(cudaFree(*data));
	        }
	        checkCudaErrors(cudaMalloc(data, size*sizeof(value_type)));
	    }
	    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer& layer, int n, int c, int h, int w, value_type *data) {
	        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
	                                                tensorFormat,
	                                                dataType,
	                                                n, c,
	                                                h,
	                                                w));
	        value_type alpha = value_type(1);
	        value_type beta  = value_type(1);
	        checkCUDNN(cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
	                                      &alpha, biasTensorDesc,
	                                      layer.d_bias,
	                                      &beta,
	                                      dstTensorDesc,
	                                      data));
	    }
	    void convoluteForward(const Layer& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData) {
        cudnnConvolutionFwdAlgo_t algo;

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                              dataType,
                                              conv.outputs,
                                              conv.inputs, 
                                              conv.kernelDim,
                                              conv.kernelDim));
 
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                    0,0, // padding
                                                    1,1, // stride
                                                    conv.stride,conv.stride, // upscale
                                                    CUDNN_CROSS_CORRELATION));	//OR CUDNN_CONVOLUTION
        // find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                &n, &c, &h, &w));

        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w));
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                0,
                                                &algo
                                                ));
        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes));
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.d_data,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        //addBias(dstTensorDesc, conv, c, *dstData); THIS CALL TO BE UNDERSTOOD AND CHANGED
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w));
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN(cudnnActivationForward(cudnnHandle,
                                            CUDNN_ACTIVATION_SIGMOID,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData));    
    }
    void convoluteBacwardData(const Layer& conv,
                          int& nI, int& cI, int& hI, int& wI,
                          value_type* diffData,
                          int& nO, int& cO, int& hO, int& wO,
                          value_type** gradData) {
    	resize(nO*cO*hO*wO, gradData);
    	checkCUDNN(cudnnSetTensor4dDescriptor(diffTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                nI, cI,
                                                hI, wI));
    	checkCUDNN(cudnnSetTensor4dDescriptor(dataGradTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                nO, cO,
                                                hO, wO));
    	 checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                              dataType,
                                              conv.outputs,
                                              conv.inputs, 
                                              conv.kernelDim,
                                              conv.kernelDim));
 
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                    0,0, // padding
                                                    1,1, // stride
                                                    conv.stride,conv.stride, // upscale
                                                    CUDNN_CROSS_CORRELATION));	//OR CUDNN_CONVOLUTION
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
        										&alpha,
        										filterDesc,
        										conv.d_data,
        										diffTensorDesc,
        										diffData,
        										convDesc,
        										&beta,
        										dataGradTensorDesc,
        										*gradData));
        nI = nO;
        cI = cO;
        hI = hO;
        wI = wO;
    }
    void convoluteBacwardFilter(const Layer& conv,
                          int& nI, int& cI, int& hI, int& wI,
                          value_type* srcData,
                          int& nO, int& cO, int& hO, int& wO,
                          value_type* diffData, value_type**gradData) {

    	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                nI, cI,
                                                hI, wI));
    	checkCUDNN(cudnnSetTensor4dDescriptor(diffTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                nO, cO,
                                                hO, wO));
    	 checkCUDNN(cudnnSetFilter4dDescriptor(filterGradDesc,
                                              dataType,
                                              conv.outputs,
                                              conv.inputs, 
                                              conv.kernelDim,
                                              conv.kernelDim));
 
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                    0,0, // padding
                                                    1,1, // stride
                                                    conv.stride,conv.stride, // upscale
                                                    CUDNN_CROSS_CORRELATION));	//OR CUDNN_CONVOLUTION
        value_type alpha = value_type(1);
        value_type beta  = value_type(1);	//accumulate filter gradients
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
        										&alpha,
        										srcTensorDesc,
        										srcData,
        										diffTensorDesc,
        										diffData,
        										convDesc,
        										&beta,
        										filterGradDesc,
        										*gradData));
    }
    void activationBackward(int& n, int& c, int& h, int& w,
                          value_type* srcData,
                          value_type* diffData, value_type* dstData, value_type**gradData) {

    	resize(n*c*h*w, gradData);
    	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w));
    	checkCUDNN(cudnnSetTensor4dDescriptor(diffTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w));
    	checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w));
    	checkCUDNN(cudnnSetTensor4dDescriptor(dataGradTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w));
 
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN(cudnnActivationBackward(cudnnHandle,
        										CUDNN_ACTIVATION_SIGMOID,
        										&alpha,
        										srcTensorDesc,
        										srcData,
        										diffTensorDesc,
        										diffData,
        										dstTensorDesc,
        										dstData,
        										&beta,
        										dataGradTensorDesc,
        										*gradData));
    }
};

struct LayerDim{
	int x, y, z, w;
};

class CNN {
	private:
		int numNNLayer;
		int numFltrLayer;
		Layer **fltrLyr;
		LayerDim *nnLayerDim;
		float learnRate;
		Network *network;
		value_type *d_nn;
		value_type *qVals;
		value_type gamma;
		int miniBatchSize;
		int firstNNLayerUnits;
		int lastNNLayerUnits;
		int totalNNUnits;
		int totalFltrUnits;
	public:
		CNN(std::string x, float z, float gamma_) {
			std::ifstream nnConfig(x.c_str());
			nnConfig >> numNNLayer;
			
			numFltrLayer = numNNLayer - 1;
			fltrLyr = new Layer*[numFltrLayer];
			nnLayerDim = new LayerDim[numNNLayer];

			nnConfig >> nnLayerDim[0].x >> nnLayerDim[0].y >> nnLayerDim[0].z >> nnLayerDim[0].w;
			miniBatchSize = nnLayerDim[0].w;

			totalFltrUnits = 0;
			for(int i = 0; i < numFltrLayer; ++i) {
				int in, out, ker, stride;
				float irD, irB;
				nnConfig >> in >> out >> ker >> stride >> irD >> irB;
				totalFltrUnits += in*out*ker*ker;
				fltrLyr[i] = new Layer(in, out, ker, stride, irD, irB);
			}
			nnConfig.close();
			learnRate = z;
			gamma = gamma_;
			network = new Network();

			d_nn = NULL;
			#ifdef TEST
				std::cout << "CNN Constructor: done!" << std::endl;
			#endif
		}
		~CNN() {
			for(int i = 0; i < numFltrLayer; ++i) {
				delete fltrLyr[i];
			}
			delete[] fltrLyr;
			delete network;
			delete[] nnLayerDim;
			delete[] qVals;
			checkCudaErrors(cudaFree(d_nn));
		}
		void init() {
			initLayers();
			forwardPropToGetDim();
			allocateNNMem();
			#ifdef TEST
				printGenAttr();
				printNNLayerDim();
				printFltrLayerAttr();
			#endif
		}
		void initLayers() {
			for(int i = 0; i < numFltrLayer; ++i) {
				fltrLyr[i]->init();
			}
			#ifdef TEST
				std::cout << "CNN initLayers: done!" << std::endl;
			#endif
		}
		void forwardPropToGetDim() {
			value_type *dstData = NULL, *srcData = NULL;
			int n = nnLayerDim[0].w, c = nnLayerDim[0].z, h = nnLayerDim[0].x, w = nnLayerDim[0].y;

			int inputSize = n*c*h*w;
			checkCudaErrors(cudaMalloc(&srcData, inputSize*sizeof(value_type)));
       		checkCudaErrors(cudaMemset(srcData, 0, inputSize*sizeof(value_type)));	//ZERO MEMSET

       		for(int i = 0; i < numFltrLayer; ++i) {
       			network->convoluteForward(*fltrLyr[i], n, c, h, w, srcData, &dstData);
       			nnLayerDim[i+1].w = n;
       			nnLayerDim[i+1].z = c;
       			nnLayerDim[i+1].x = h;
       			nnLayerDim[i+1].y = w;
       			network->activationForward(n, c, h, w, dstData, &srcData);
       		}
       		#ifdef TEST
       			std::cout << "Resulting Weights: " << std::endl;
       			printDeviceVector(n*c*h*w, srcData);
       		#endif
       		checkCudaErrors(cudaFree(srcData));
        	checkCudaErrors(cudaFree(dstData));
        	#ifdef TEST
				std::cout << "CNN forwardPropToGetDim: done!" << std::endl;
			#endif
		}
		//has to be called after dimensions are known
		void allocateNNMem() {
			totalNNUnits = 0;
			firstNNLayerUnits = nnLayerDim[0].x*nnLayerDim[0].y*nnLayerDim[0].z*nnLayerDim[0].w;
			for(int i = 0; i < numNNLayer; ++i) {
				int temp = nnLayerDim[i].x*nnLayerDim[i].y*nnLayerDim[i].z*nnLayerDim[i].w;
				totalNNUnits += temp;
				if(i == numNNLayer - 1)
					lastNNLayerUnits = temp;
			}
			checkCudaErrors(cudaMalloc(&d_nn, totalNNUnits*sizeof(value_type)));
			qVals = new value_type[lastNNLayerUnits];
			#ifdef TEST
				std::cout << "CNN allocateNNMem: done!" << std::endl;
			#endif
		}
		//h_inpLayer must have firstNNLayerUnits in it
		void forwardProp(value_type *h_inpLayer) {
			value_type *dstData = NULL, *srcData = NULL;
			int n = nnLayerDim[0].w, c = nnLayerDim[0].z, h = nnLayerDim[0].x, w = nnLayerDim[0].y;
			
			int inputSize = n*c*h*w;
			assert(inputSize == firstNNLayerUnits);
			checkCudaErrors(cudaMalloc(&srcData, inputSize*sizeof(value_type)) );
       		checkCudaErrors(cudaMemcpyHTD(srcData, h_inpLayer, inputSize*sizeof(value_type)));
       		//copy to d_nn
       		checkCudaErrors(cudaMemcpyHTD(d_nn, h_inpLayer, inputSize*sizeof(value_type)));

       		int tnnu = inputSize;
			for(int i = 0; i < numFltrLayer; ++i) {
				network->convoluteForward(*fltrLyr[i], n, c, h, w, srcData, &dstData);
				//may be some different modes and
				// no activation at some points 
				// utility can be added here
				network->activationForward(n, c, h, w, dstData, &srcData);
				//cpy to d_nn
				assert(n*c*h*w == nnLayerDim[i+1].x*nnLayerDim[i+1].y*nnLayerDim[i+1].z*nnLayerDim[i+1].w);
				checkCudaErrors(cudaMemcpyDTD(d_nn + tnnu, srcData, n*c*h*w*sizeof(value_type)));
				tnnu += n*c*h*w;
			}
			assert(tnnu == totalNNUnits);
			assert(n*c*h*w == lastNNLayerUnits);

	        checkCudaErrors(cudaMemcpy(qVals, srcData, lastNNLayerUnits*sizeof(value_type), cudaMemcpyDeviceToHost));
			
			#ifdef TEST
				printDeviceVector(n*c*h*w, srcData);
			#endif

			checkCudaErrors(cudaFree(srcData));
        	checkCudaErrors(cudaFree(dstData));
        	#ifdef TEST
				std::cout << "CNN forwardProp: done!" << std::endl;
			#endif
		}

		void backwardProp(value_type *h_err) {
			value_type *diffData = NULL, *gradData = NULL;
			int n = nnLayerDim[numNNLayer-1].w, c = nnLayerDim[numNNLayer-1].z, h = nnLayerDim[numNNLayer-1].x, w = nnLayerDim[numNNLayer-1].y;
			int nI, cI, hI, wI;
			int inputSize = n*c*h*w;
			assert(inputSize == lastNNLayerUnits);
			checkCudaErrors(cudaMalloc(&diffData, inputSize*sizeof(value_type)) );
       		checkCudaErrors(cudaMemcpyHTD(diffData, h_err, inputSize*sizeof(value_type)));

       		//reset all fltr layer gradient
       		for(int i = 0; i < numFltrLayer; ++i)
       			fltrLyr[i]->resetGrad();

       		int tnnu = totalNNUnits - inputSize;
       		for(int i = numFltrLayer; i >= 1; --i) {
       			nI = nnLayerDim[i-1].w, cI = nnLayerDim[i-1].z, hI = nnLayerDim[i-1].x, wI = nnLayerDim[i-1].y;
       			network->activationBackward(n, c, h, w, d_nn + tnnu, diffData, d_nn + tnnu, &gradData);
       			network->convoluteBacwardFilter(*fltrLyr[i-1], nI, cI, hI, wI, d_nn + tnnu - nI*cI*hI*wI, n, c, h, w, gradData, &(fltrLyr[i-1]->d_grad));
       			if(i > 1) {
       				network->convoluteBacwardData(*fltrLyr[i-1], n, c, h, w, gradData, nI, cI, hI, wI, &diffData);
	       			tnnu -= n*c*h*w;	//here n,c,h,w <- nI, cI, hI, wI
	       			assert(n==nI && c==cI && h==hI && w==wI);
	       		}
       			
       		}
       		assert(tnnu == firstNNLayerUnits);
       		//update layers
       		for(int i = 0; i < numFltrLayer; ++i)
       			fltrLyr[i]->update(learnRate, gamma, miniBatchSize);
       		checkCudaErrors(cudaFree(diffData));
        	checkCudaErrors(cudaFree(gradData));
		}

		int argMaxQVal(int numAction) {
			assert(numAction == nnLayerDim[numNNLayer-1].z);
			int idx = 0;
			for(int i = 0; i < numAction; ++i) {
				if(qVals[idx] < qVals[i])
					idx = i;
			}
			return idx;
		}

		value_type* getQVals() {
			return qVals;
		}

		void printGenAttr() {
			std::cout << "Total NN Units: " << totalNNUnits << std::endl;
			std::cout << "Total Fltr Units: " << totalFltrUnits << std::endl;
			std::cout << "First layer NN Units: " << firstNNLayerUnits << std::endl;
			std::cout << "Last layer NN Units: " << lastNNLayerUnits << std::endl;
			std::cout << "Num NN Layers: " << numNNLayer << std::endl;
			std::cout << "Num fltr layers: " << numFltrLayer << std::endl;
			std::cout << "Learning Rate: " << learnRate << std::endl;
		}

		void printNNLayerDim() {
			std::cout << "NN Layer Dimensions" << std::endl;
			for (int i = 0; i < numNNLayer; ++i) {
				int n = nnLayerDim[i].w, c = nnLayerDim[i].z, h = nnLayerDim[i].x, w = nnLayerDim[i].y;
				std::cout << "w z x y == n c h w: " << i << " layer " << n << " " << c << " " << h << " " << w << std::endl;
			}
			#ifdef TEST
				std::cout << "CNN nnLayerDim: done!" << std::endl;
			#endif
		}

		void printFltrLayerAttr() {
			std::cout << "Filter Layer Attributes" << std::endl;
			for (int i = 0; i < numFltrLayer; ++i) {
				int k = fltrLyr[i]->outputs, c = fltrLyr[i]->inputs, c1 = fltrLyr[i]->kernelDim, h = fltrLyr[i]->stride;
				float w = fltrLyr[i]->iRangeD, u = fltrLyr[i]->iRangeB;
				std::cout << "out in ker stride rangeD range B: " << i << " layer " << k << " " << c << " " << c1 << " " <<  h << " " << w << " " << u << std::endl;
			}
			#ifdef TEST
				std::cout << "CNN nnLayerDim: done!" << std::endl;
			#endif
		}

		void printFltrLayer(int i) {
			std::cout << "Filter Layer " << i << std::endl;
			assert(i < numFltrLayer);
			printDeviceVector(fltrLyr[i]->inputs*fltrLyr[i]->outputs*fltrLyr[i]->kernelDim*fltrLyr[i]->kernelDim, fltrLyr[i]->d_data);
		}

		void printAllFltrLayer() {
			for(int i = 0; i < numFltrLayer; ++i) {
				printFltrLayer(i);
			}
		}

		void printFltrLayerGrad(int i) {
			std::cout << "Filter Layer Grad " << i << std::endl;
			assert(i < numFltrLayer);
			printDeviceVector(fltrLyr[i]->inputs*fltrLyr[i]->outputs*fltrLyr[i]->kernelDim*fltrLyr[i]->kernelDim, fltrLyr[i]->d_grad);
		}

		void printAllFltrLayerGrad() {
			for(int i = 0; i < numFltrLayer; ++i) {
				printFltrLayerGrad(i);
			}
		}

		void testForwardAndBackward() {
			int inputSize = nnLayerDim[0].x*nnLayerDim[0].y*nnLayerDim[0].z*nnLayerDim[0].w;
			value_type *testInput = new value_type[inputSize];
			for(int i = 0; i < inputSize; ++i) {
				testInput[i] = ((value_type)(rand()))/((value_type)(RAND_MAX));
			}
			std::cout << "Test Input" << std::endl;
			printHostVector(inputSize, testInput);
			std::cout << "Filter Layers" << std::endl;
			printAllFltrLayer();
			forwardProp(testInput);
			std::cout << "argMaxQVal is: " << argMaxQVal(nnLayerDim[numNNLayer-1].z) << std::endl;

			std::cout << "Backpropagation Started: " << std::endl;
			value_type *err = new value_type[lastNNLayerUnits];
			for(int i = 0; i < lastNNLayerUnits; ++i) {
				err[i] = -qVals[i]/2.0;
			}
			std::cout << "Errors: " << std::endl;
			printHostVector(lastNNLayerUnits, err);
			backwardProp(err);
			std::cout << "Fltr Layers gradients " << std::endl;
			printAllFltrLayerGrad();
			std::cout << "Fltr Layers after backpropagation" << std::endl;
			printAllFltrLayer();
			delete[] testInput;
			delete[] err;
		}
};

int main() {
CNN cnn("nnConfigTest", 0.3, 0.1);
cnn.init();
cnn.testForwardAndBackward();
return 0;
}



//ACTIVATION_RELU not working ... WHYYYYYYYYYYYYYYYYYYYYYYY


