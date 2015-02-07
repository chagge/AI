//cnn3.cu
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include "util.h"
#include "layer.h"
#include "network.h"
#include "cnn3.h"
#include <cmath>
#include "info.h"

//#define CNNTESTITER

CNN::CNN(Info x) {
	info = x;
	
	std::ifstream nnConfig(info.nnConfig.c_str());
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

	nnConfig >> stepSize;
	nnConfig >> baseLearnRate;
	nnConfig >> dropFactor;
	nnConfig >> rho;
	nnConfig >> negSlopeRelu;
	nnConfig.close();

	network = new Network();

	saveFltrCntr = 0;
	numSteps = 0;
	d_nn = NULL;
	qVals = NULL;

	init();
}

CNN::~CNN() {
	for(int i = 0; i < numFltrLayer; ++i)
		delete fltrLyr[i];
	delete[] fltrLyr;
	delete network;
	delete[] nnLayerDim;
	delete[] qVals;
	checkCudaErrors(cudaFree(d_nn));
}

void CNN::init() {
	initLayers();
	forwardPropToGetDim();
	allocateNNMem();
	if(info.debugCNN) {
		cnnLog.open(info.cnnLogFile.c_str());
		printGenAttr();
		printNNLayerDim();
		printFltrLayerAttr();
	}
}

void CNN::initLayers() {
	for(int i = 0; i < numFltrLayer; ++i) {
		fltrLyr[i]->init();
	}
}

void CNN::forwardPropToGetDim() {
	value_type *dstData = NULL, *srcData = NULL;
	int n = nnLayerDim[0].w, c = nnLayerDim[0].z, h = nnLayerDim[0].x, w = nnLayerDim[0].y;

	int inputSize = n*c*h*w;
	checkCudaErrors(cudaMalloc(&srcData, inputSize*sizeof(value_type)));
	checkCudaErrors(cudaMemset(srcData, 0, inputSize*sizeof(value_type)));	//ZERO MEMSET

	for(int i = 0; i < numFltrLayer; ++i) {
		network->convoluteForward(*fltrLyr[i], n, c, h, w, srcData, &dstData, false);
		nnLayerDim[i+1].w = n; nnLayerDim[i+1].z = c; nnLayerDim[i+1].x = h; nnLayerDim[i+1].y = w;
		network->activationForward(n, c, h, w, dstData, &srcData);
	}
	checkCudaErrors(cudaFree(srcData));
	checkCudaErrors(cudaFree(dstData));
}

//has to be called after dimensions are known
void CNN::allocateNNMem() {
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
}

void CNN::printGenAttr() {
	cnnLog << "Total NN Units: " << totalNNUnits << std::endl;
	cnnLog << "Total Fltr Units: " << totalFltrUnits << std::endl;
	cnnLog << "First layer NN Units: " << firstNNLayerUnits << std::endl;
	cnnLog << "Last layer NN Units: " << lastNNLayerUnits << std::endl;
	cnnLog << "Num NN Layers: " << numNNLayer << std::endl;
	cnnLog << "Num fltr layers: " << numFltrLayer << std::endl;
	cnnLog << "Base Learning Rate: " << baseLearnRate << std::endl;
	cnnLog << "Step size i.e. drop LR after: " << stepSize << std::endl;
	cnnLog << "Drop factor: " << dropFactor << std::endl;
	cnnLog << "Rho : " << rho << std::endl;
}

void CNN::printNNLayerDim() {
	cnnLog << "NN Layer Dimensions: " << std::endl;
	for (int i = 0; i < numNNLayer; ++i) {
		int n = nnLayerDim[i].w, c = nnLayerDim[i].z, h = nnLayerDim[i].x, w = nnLayerDim[i].y;
		cnnLog << "w z x y == n c h w: " << i << "th layer: " << n << " " << c << " " << h << " " << w << std::endl;
	}
}

void CNN::printFltrLayerAttr() {
	cnnLog << "Filter Layer Attributes: " << std::endl;
	for (int i = 0; i < numFltrLayer; ++i) {
		int k = fltrLyr[i]->outputs, c = fltrLyr[i]->inputs, c1 = fltrLyr[i]->kernelDim, h = fltrLyr[i]->stride;
		float w = fltrLyr[i]->iRangeD, u = fltrLyr[i]->iRangeB;
		cnnLog << "outputs inputs kernelSz stride rangeD rangeB: " << i << " layer " << k << " " << c << " " << c1 << " " <<  h << " " << w << " " << u << std::endl;
	}
}

//h_inpLayer must have firstNNLayerUnits in it
void CNN::forwardProp(value_type *h_inpLayer) {
	value_type *dstData = NULL, *srcData = NULL;
	int n = nnLayerDim[0].w, c = nnLayerDim[0].z, h = nnLayerDim[0].x, w = nnLayerDim[0].y;
	
	int inputSize = n*c*h*w;
	assert(inputSize == firstNNLayerUnits);
	checkCudaErrors(cudaMalloc(&srcData, inputSize*sizeof(value_type)));
	checkCudaErrors(cudaMemcpyHTD(srcData, h_inpLayer, inputSize*sizeof(value_type)));
	//copy to d_nn
	checkCudaErrors(cudaMemcpyHTD(d_nn, h_inpLayer, inputSize*sizeof(value_type)));

	int tnnu = inputSize;
	for(int i = 0; i < numFltrLayer; ++i) {
		network->convoluteForward(*fltrLyr[i], n, c, h, w, srcData, &dstData, info.isBias);
		network->activationForwardLeakyRELU(n, c, h, w, dstData, &srcData, negSlopeRelu);

		assert(n*c*h*w == nnLayerDim[i+1].x*nnLayerDim[i+1].y*nnLayerDim[i+1].z*nnLayerDim[i+1].w);
		//cpy to d_nn
		checkCudaErrors(cudaMemcpyDTD(d_nn + tnnu, srcData, n*c*h*w*sizeof(value_type)));
		tnnu += n*c*h*w;
	}
	assert(tnnu == totalNNUnits);
	assert(n*c*h*w == lastNNLayerUnits);

    checkCudaErrors(cudaMemcpyDTH(qVals, srcData, lastNNLayerUnits*sizeof(value_type)));
    if(info.debugCNN) {
		cnnLog << "qVals: " << std::endl;
		printHostVectorInFile(lastNNLayerUnits, qVals, cnnLog);
	}

	checkCudaErrors(cudaFree(srcData));
	checkCudaErrors(cudaFree(dstData));
}

void CNN::backwardProp(value_type *h_err) {
	value_type *diffData = NULL, *gradData = NULL;
	int n = nnLayerDim[numNNLayer-1].w, c = nnLayerDim[numNNLayer-1].z, h = nnLayerDim[numNNLayer-1].x, w = nnLayerDim[numNNLayer-1].y;
	int nI, cI, hI, wI;
	int inputSize = n*c*h*w;
	
	assert(inputSize == lastNNLayerUnits);
	
	checkCudaErrors(cudaMalloc(&diffData, inputSize*sizeof(value_type)));
	checkCudaErrors(cudaMemcpyHTD(diffData, h_err, inputSize*sizeof(value_type)));

	//reset all fltr layer gradient
	for(int i = 0; i < numFltrLayer; ++i)
		fltrLyr[i]->resetGrad();

	int tnnu = totalNNUnits - inputSize;
	for(int i = numFltrLayer; i >= 1; --i) {
		nI = nnLayerDim[i-1].w, cI = nnLayerDim[i-1].z, hI = nnLayerDim[i-1].x, wI = nnLayerDim[i-1].y;
		if(info.isBias)
			network->convoluteBackwardBias(n, c, h, w, 
											diffData, 
											&(fltrLyr[i-1]->d_grad_bias));
		network->activationBackwardLeakyRELU(n, c, h, w, 
										d_nn + tnnu, 
										diffData, 
										d_nn + tnnu, 
										&gradData, 
										negSlopeRelu);
		network->convoluteBacwardFilter(*fltrLyr[i-1],
										nI, cI, hI, wI, 
										d_nn + tnnu - nI*cI*hI*wI, 
										n, c, h, w, 
										gradData, 
										&(fltrLyr[i-1]->d_grad));
		if(i > 1) {
			network->convoluteBacwardData(*fltrLyr[i-1],
											n, c, h, w, 
											gradData, 
											nI, cI, hI, wI, 
											&diffData);

			tnnu -= n*c*h*w;	//here n,c,h,w <- nI, cI, hI, wI
			assert(n==nI && c==cI && h==hI && w==wI);
		}
		
	}
	assert(tnnu == firstNNLayerUnits);
	//update layers
	for(int i = 0; i < numFltrLayer; ++i)
		fltrLyr[i]->update(learnRate, rho, miniBatchSize, info.isBias);

	checkCudaErrors(cudaFree(diffData));
	checkCudaErrors(cudaFree(gradData));
}

void CNN::learn(value_type *h_inpLayer, value_type *target, int mIter) {
	learnRate = baseLearnRate;
	prevLoss = -1.0f;
	for(int i = 0; i < mIter; ++i) {
		if(info.debugCNN) {
			cnnLog << "Step: " << i << std::endl;
		}
		step(h_inpLayer, target);
	}
	if(numSteps%stepSize==0) {
		baseLearnRate = baseLearnRate * dropFactor;
	}
}

void CNN::step(value_type *h_inpLayer, value_type *target) {
	resetNN();
	resetQVals();

	forwardProp(h_inpLayer);

	value_type *err = new value_type[lastNNLayerUnits];
	loss = 0;
	for(int i = 0; i < lastNNLayerUnits; ++i) {
		err[i] = -1*(target[i] - qVals[i]);
		loss += err[i]*err[i];
	}
	loss = (1.0/miniBatchSize)*std::sqrt(loss);

	if(info.debugCNN)
		cnnLog << "Loss: " << loss << " Opt loss till: " << prevLoss << std::endl;

	if(prevLoss == -1.0f || (prevLoss >= loss)) {
		prevLoss = loss;
		copyFltrToHist();
		backwardProp(err);
		if(info.debugCNN)
			cnnLog << "Backpropagated error." << std::endl;			
	} else {
		learnRate = learnRate * dropFactor;
		loadFltrFrmHist();
		if(info.debugCNN)
			cnnLog << "NO Backpropagation. Learning Rate Decreased." << std::endl;
	}
	numSteps++;
	delete[] err;
}

void CNN::loadFltrFrmHist() {
	for(int i = 0; i < numFltrLayer; ++i) {
		fltrLyr[i]->copyDataDHTD();
	}
}
void CNN::copyFltrToHist() {
	for(int i = 0; i < numFltrLayer; ++i) {
		fltrLyr[i]->copyDataDTDH();
	}
}

int CNN::chooseAction(value_type *h_inpLayer, int numAction) {
	resetNN();
	resetQVals();
	forwardProp(h_inpLayer);
	int ret = argMaxQVal(numAction);
	resetNN();
	return ret;
}

int CNN::argMaxQVal(int numAction) {
	assert(numAction == nnLayerDim[numNNLayer-1].z);
	
	int idx = 0;
	for(int i = 0; i < numAction; ++i) {
		if(qVals[idx] < qVals[i])
			idx = i;
	}
	return idx;
}

value_type* CNN::forwardNGetQVal(value_type *h_inpLayer) {
	resetNN();
	resetQVals();
	forwardProp(h_inpLayer);
	resetNN();
	return getQVals();
}

value_type* CNN::getQVals() {
	return qVals;
}

void CNN::resetNN() {
	checkCudaErrors(cudaMemset(d_nn, 0, totalNNUnits*sizeof(value_type)));
}

void CNN::resetQVals() {
	memset(qVals, 0, lastNNLayerUnits*sizeof(value_type));
}

void CNN::testIterate(int numIter) {
	int inputSize = nnLayerDim[0].x*nnLayerDim[0].y*nnLayerDim[0].z*nnLayerDim[0].w;
	value_type *testInput = new value_type[inputSize];
	for(int i = 0; i < inputSize; ++i) {
		testInput[i] = value_type(rand_normal(0, 1));
	}
	value_type *targ = new value_type[lastNNLayerUnits];
	memset(targ, 0, lastNNLayerUnits*sizeof(float));
	for(int i = 0; i < nnLayerDim[numNNLayer-1].w; ++i) {
		int j_ = rand()%nnLayerDim[numNNLayer-1].z;
		targ[i*nnLayerDim[numNNLayer-1].z+j_] = 1.0f;
	}

	learn(testInput, targ, numIter);
	if(info.debugCNN) {
		cnnLog << "Target: " << std::endl;
		printHostVectorInFile(lastNNLayerUnits, targ, cnnLog);
	}

	delete[] targ;
	delete[] testInput;
}

void CNN::saveFilterWts() {
	std::string path = info.dataPath + "/WF" + toString(saveFltrCntr);
	std::ofstream myFile(path.c_str());
	for(int i = 0; i < numFltrLayer; ++i)
		fltrLyr[i]->copyDataDTH();
	for(int i = 0; i < numFltrLayer; ++i) {
		int size = fltrLyr[i]->inputs*fltrLyr[i]->outputs*fltrLyr[i]->kernelDim*fltrLyr[i]->kernelDim;
		for(int j = 0; j < size; ++j) {
			myFile << fltrLyr[i]->h_data[j] << " ";
		}
		myFile << std::endl;
	}
	myFile.close();
	saveFltrCntr++;
}

int CNN::getInputFMSize() {
	return nnLayerDim[0].x*nnLayerDim[0].y;
}

int CNN::getCurWFNum() {
	return saveFltrCntr;
}

float CNN::getCurrentLoss() {
	return loss;
}

float CNN::getPrevLoss() {
	return prevLoss;
}
