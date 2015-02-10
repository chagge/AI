//util.h
#ifndef __UTIL_H__
#define __UTIL_H__ 1

#define UNDEF -3
#define FAIL 0
#define SUCCESS 1

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <fstream>


#define ALE_HEIGHT 210
#define ALE_WIDTH 160

typedef float value_type;


#define cudaMemcpyHTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost)
#define cudaMemcpyDTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice)


#define NBDIMS 4
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

struct History{
	int fiJ, reward, act, isTerm, fiJN;
};

struct LayerDim{
	int x, y, z, w;
};
inline int toInt(std::string s) {int i;std::stringstream(s)>>i;return i;}
inline std::string toString(int i) {std::string s;std::stringstream ss;ss<<i;ss>>s;return s;}

char *inputString(FILE*, size_t);

double hex2int(char, char);
int ntsc2gray(char, char);
double rand_normal(double, double);
void printDeviceVector(int, value_type*);
void printHostVector(int, value_type*);
void printDeviceVectorInFile(int, value_type*, std::ofstream&);
void printHostVectorInFile(int, value_type*, std::ofstream&, std::string x = " ");

#endif
