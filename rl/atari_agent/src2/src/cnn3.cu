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

//#define CNNTESTITER

CNN::CNN(std::string x, float z, float gamma_, std::string dataPath_) {
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
	nnConfig >> stepSize;
	nnConfig.close();
	baseLearnRate = z;
	gamma = gamma_;
	dataPath = dataPath_;
	network = new Network();
	saveFltrCntr = 0;
	dropFactor = 0.1f;
	numSteps = 0;


	d_nn = NULL;
	#ifdef TEST
		std::cout << "CNN Constructor: done!" << std::endl;
	#endif
}
CNN::~CNN() {
	
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
	#ifdef TEST
		printGenAttr();
		printNNLayerDim();
		printFltrLayerAttr();
	#endif
}
void CNN::initLayers() {
	for(int i = 0; i < numFltrLayer; ++i) {
		fltrLyr[i]->init();
	}
	#ifdef TEST
		std::cout << "CNN initLayers: done!" << std::endl;
	#endif
}
void CNN::forwardPropToGetDim() {
	value_type *dstData = NULL, *srcData = NULL;
	int n = nnLayerDim[0].w, c = nnLayerDim[0].z, h = nnLayerDim[0].x, w = nnLayerDim[0].y;

	int inputSize = n*c*h*w;
	checkCudaErrors(cudaMalloc(&srcData, inputSize*sizeof(value_type)));
		checkCudaErrors(cudaMemset(srcData, 0, inputSize*sizeof(value_type)));	//ZERO MEMSET

	for(int i = 0; i < numFltrLayer; ++i) {
		network->convoluteForward(*fltrLyr[i], n, c, h, w, srcData, &dstData, false);
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
	#ifdef TEST
		std::cout << "CNN allocateNNMem: done!" << std::endl;
	#endif
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
		network->convoluteForward(*fltrLyr[i], n, c, h, w, srcData, &dstData, true);
		//may be some different modes and
		// no activation at some points 
		// utility can be added here
		network->activationForwardLeakyRELU(n, c, h, w, dstData, &srcData, 0.01f);
		//cpy to d_nn
		assert(n*c*h*w == nnLayerDim[i+1].x*nnLayerDim[i+1].y*nnLayerDim[i+1].z*nnLayerDim[i+1].w);
		checkCudaErrors(cudaMemcpyDTD(d_nn + tnnu, srcData, n*c*h*w*sizeof(value_type)));
		tnnu += n*c*h*w;
	}
	assert(tnnu == totalNNUnits);
	assert(n*c*h*w == lastNNLayerUnits);

    checkCudaErrors(cudaMemcpyDTH(qVals, srcData, lastNNLayerUnits*sizeof(value_type)));
	
	#ifdef TEST
		printDeviceVector(n*c*h*w, srcData);
	#endif

	checkCudaErrors(cudaFree(srcData));
	checkCudaErrors(cudaFree(dstData));
	#ifdef TEST
		std::cout << "CNN forwardProp: done!" << std::endl;
	#endif
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
		network->convoluteBackwardBias(n, c, h, w, diffData, &(fltrLyr[i-1]->d_grad_bias));
		network->activationBackwardLeakyRELU(n, c, h, w, d_nn + tnnu, diffData, d_nn + tnnu, &gradData, 0.01f);
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

int CNN::argMaxQVal(int numAction) {
	assert(numAction == nnLayerDim[numNNLayer-1].z);
	int idx = 0;
	for(int i = 0; i < numAction; ++i) {
		if(qVals[idx] < qVals[i])
			idx = i;
	}
	return idx;
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

void CNN::printGenAttr() {
	std::cout << "Total NN Units: " << totalNNUnits << std::endl;
	std::cout << "Total Fltr Units: " << totalFltrUnits << std::endl;
	std::cout << "First layer NN Units: " << firstNNLayerUnits << std::endl;
	std::cout << "Last layer NN Units: " << lastNNLayerUnits << std::endl;
	std::cout << "Num NN Layers: " << numNNLayer << std::endl;
	std::cout << "Num fltr layers: " << numFltrLayer << std::endl;
	std::cout << "Learning Rate: " << learnRate << std::endl;
}

void CNN::printNNLayerDim() {
	std::cout << "NN Layer Dimensions" << std::endl;
	for (int i = 0; i < numNNLayer; ++i) {
		int n = nnLayerDim[i].w, c = nnLayerDim[i].z, h = nnLayerDim[i].x, w = nnLayerDim[i].y;
		std::cout << "w z x y == n c h w: " << i << " layer " << n << " " << c << " " << h << " " << w << std::endl;
	}
	#ifdef TEST
		std::cout << "CNN nnLayerDim: done!" << std::endl;
	#endif
}

void CNN::printFltrLayerAttr() {
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

void CNN::printFltrLayer(int i) {
	std::cout << "Filter Layer " << i << std::endl;
	assert(i < numFltrLayer);
	printDeviceVector(fltrLyr[i]->inputs*fltrLyr[i]->outputs*fltrLyr[i]->kernelDim*fltrLyr[i]->kernelDim, fltrLyr[i]->d_data);
}

void CNN::printAllFltrLayer() {
	for(int i = 0; i < numFltrLayer; ++i) {
		printFltrLayer(i);
	}
}

void CNN::printFltrLayerGrad(int i) {
	std::cout << "Filter Layer Grad " << i << std::endl;
	assert(i < numFltrLayer);
	printDeviceVector(fltrLyr[i]->inputs*fltrLyr[i]->outputs*fltrLyr[i]->kernelDim*fltrLyr[i]->kernelDim, fltrLyr[i]->d_grad);
}

void CNN::printAllFltrLayerGrad() {
	for(int i = 0; i < numFltrLayer; ++i) {
		printFltrLayerGrad(i);
	}
}

void CNN::testForwardAndBackward() {
	int inputSize = nnLayerDim[0].x*nnLayerDim[0].y*nnLayerDim[0].z*nnLayerDim[0].w;
	value_type *testInput = new value_type[inputSize];
	for(int i = 0; i < inputSize; ++i) {
		testInput[i] = value_type(rand_normal(0, 1));
	}
	#ifdef TESTP
		std::cout << "Test Input" << std::endl;
		printHostVector(inputSize, testInput);
		std::cout << "Filter Layers" << std::endl;
		printAllFltrLayer();
	#endif
	forwardProp(testInput);
	std::cout << "argMaxQVal in first image is: " << argMaxQVal(nnLayerDim[numNNLayer-1].z) << std::endl;

	std::cout << "Backpropagation Started: " << std::endl;
	value_type *err = new value_type[lastNNLayerUnits];
	for(int i = 0; i < lastNNLayerUnits; ++i) {
		err[i] = -qVals[i]/2.0;
	}
	#ifdef TESTP
		std::cout << "Errors: " << std::endl;
		printHostVector(lastNNLayerUnits, err);
	#endif
	backwardProp(err);
	#ifdef TESTP
		std::cout << "Fltr Layers gradients " << std::endl;
		printAllFltrLayerGrad();
		std::cout << "Fltr Layers after backpropagation" << std::endl;
		printAllFltrLayer();
	#endif
	delete[] testInput;
	delete[] err;
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

	std::cout << "Target: " << std::endl;
	printHostVector(lastNNLayerUnits, targ);
	learnRate = baseLearnRate;
	prevLoss = -1.0f;
	for(int i = 0; i < numIter; ++i) {
		std::cout << "iter no.: " << i << std::endl;
		step(testInput, targ);
	}
	printHostVector(lastNNLayerUnits, targ);
	std::cout << "Action chosen: " << std::endl;
	std::cout << chooseAction(testInput, nnLayerDim[numNNLayer-1].z) << std::endl;
	delete[] targ;
	delete[] testInput;
}

void CNN::step(value_type *h_inpLayer, value_type *target) {
	value_type *err = new value_type[lastNNLayerUnits];
	resetNN();
	resetQVals();
	forwardProp(h_inpLayer);
	loss = 0;
	for(int i = 0; i < lastNNLayerUnits; ++i) {
		err[i] = -1*(target[i] - qVals[i]);
		loss += err[i]*err[i];
	}
	loss = (1.0/miniBatchSize)*std::sqrt(loss);
	#ifdef CNNTESTITER
				std::cout << "LOSS: " << loss << " PREV LOSS: " << prevLoss << std::endl;
	#endif
	if(prevLoss == -1.0f || (prevLoss >= loss)) {
			prevLoss = loss;
			copyFltrToHist();
			backwardProp(err);
			#ifdef CNNTESTITER
				std::cout << "YES BACKPROP" << std::endl;
				printHostVector(lastNNLayerUnits, qVals);
			#endif
			
		} else {
			#ifdef CNNTESTITER
				std::cout << "NO BACKPROP" << std::endl;
				printHostVector(lastNNLayerUnits, qVals);
			#endif
			learnRate = learnRate * dropFactor;
			loadFltrFrmHist();
		}
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

void CNN::learn(value_type *h_inpLayer, value_type *target, int mIter) {
	learnRate = baseLearnRate;
	prevLoss = -1.0f;
	std::ofstream lossFile("loss.txt");
	for(int i = 0; i < mIter; ++i) {
		step(h_inpLayer, target);
		lossFile << "iter no. " << i << " loss: " << loss << " prevloss: " << prevLoss << std::endl;
		//std::cout << "iteration No." << i << " loss:" << prevLoss << std::endl;
	}
	lossFile.close();
	numSteps++;
	if(numSteps%stepSize==0) {
		baseLearnRate = baseLearnRate * dropFactor;
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

value_type* CNN::forwardNGetQVal(value_type *h_inpLayer) {
	resetNN();
	resetQVals();
	forwardProp(h_inpLayer);
	resetNN();
	return getQVals();
}

void CNN::saveFilterWts() {
	std::string path = dataPath + "/WF" + toString(saveFltrCntr);
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
