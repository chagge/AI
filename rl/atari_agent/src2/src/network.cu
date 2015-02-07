//network.cu
#include "network.h"
#include "cudnn.h"
#include "cudnn.h"
    
__global__ void leakyReluActivateForward(value_type *d_in, value_type *d_out, int n, value_type slope) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= n)
    return;
  d_out[idx] = d_in[idx];
  if(d_in[idx]<=0)
    d_out[idx] *= slope;
}

__global__ void leakyReluActivateBackward(value_type *d_in, value_type *d_out, value_type *d_z, int n, value_type slope) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= n)
    return;
  d_out[idx] = d_in[idx];
  if(d_z[idx]<=0)
    d_out[idx] *= slope;
}

void Network::createHandles() {
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

void Network::destroyHandles() {
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


Network::Network() {
  dataType = CUDNN_DATA_FLOAT;
  tensorFormat = CUDNN_TENSOR_NCHW;
  createHandles();
}

Network::~Network() {
  destroyHandles();
}
void Network::resize(int size, value_type **data) {
    if(*data != NULL)
    {
        checkCudaErrors(cudaFree(*data));
    }
    checkCudaErrors(cudaMalloc(data, size*sizeof(value_type)));
}
void Network::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer& layer, int n, int c, int h, int w, value_type *data) {
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
void Network::convoluteForward(const Layer& conv,
                  int& n, int& c, int& h, int& w,
                  value_type* srcData, value_type** dstData, bool biasAdd) {
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
                                                conv.stride,conv.stride, // stride
                                                1,1, // upscale
                                                CUDNN_CROSS_CORRELATION));  //OR CUDNN_CONVOLUTION
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
      checkCudaErrors(cudaMalloc(&workSpace,sizeInBytes));
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
    if(biasAdd)
      addBias(dstTensorDesc, conv, 1, c, 1, 1, *dstData);
    if (sizeInBytes!=0)
    {
      checkCudaErrors(cudaFree(workSpace));
    }
}
void Network::activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
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
                                        CUDNN_ACTIVATION_RELU,
                                        &alpha,
                                        srcTensorDesc,
                                        srcData,
                                        &beta,
                                        dstTensorDesc,
                                        *dstData));    
}
void Network::activationForwardLeakyRELU(int n, int c, int h, int w, value_type* srcData, value_type** dstData, value_type slope) {
  resize(n*c*h*w, dstData);
  dim3 threadsPerBlock(BLOCKSIZE);
  dim3 numBlocks((n*c*h*w-1)/threadsPerBlock.x + 1);
  leakyReluActivateForward<<<numBlocks, threadsPerBlock>>>(srcData, *dstData, n*c*h*w, slope);
}
void Network::convoluteBacwardData(const Layer& conv,
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
                                                conv.stride,conv.stride, // stride
                                                1,1, // upscale
                                                CUDNN_CROSS_CORRELATION));  //OR CUDNN_CONVOLUTION
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
void Network::convoluteBacwardFilter(const Layer& conv,
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
                                                conv.stride,conv.stride, // stride
                                                1,1, // upscale
                                                CUDNN_CROSS_CORRELATION));  //OR CUDNN_CONVOLUTION
    value_type alpha = value_type(1);
    value_type beta  = value_type(1); //accumulate filter gradients
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
void Network::activationBackward(int& n, int& c, int& h, int& w,
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
                        CUDNN_ACTIVATION_RELU,
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

void Network::activationBackwardLeakyRELU(int& n, int& c, int& h, int& w,
                      value_type* srcData,
                      value_type* diffData, value_type* dstData, value_type**gradData, value_type slope) {
  resize(n*c*h*w, gradData);
  dim3 threadsPerBlock(BLOCKSIZE);
  dim3 numBlocks((n*c*h*w-1)/threadsPerBlock.x + 1);
  leakyReluActivateBackward<<<numBlocks, threadsPerBlock>>>(diffData, *gradData, srcData, n*c*h*w, slope);
}

void Network::convoluteBackwardBias(int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type**gradData) {
  resize(n*c*h*w, gradData);
  checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h, w));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
                                            tensorFormat,
                                            dataType,
                                            1, c,
                                            1, 1));
  value_type alpha = value_type(1);
  value_type beta  = value_type(1); //accumulate beta gradients
  checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          biasTensorDesc,
                                          *gradData));
}