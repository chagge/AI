//cnn3.h
#ifndef __CNN3_H__
#define __CNN3_H__

#include "util.h"
#include "layer.h"
#include "network.h"
#include <string>
#include "info.h"
#include <fstream>

class CNN {
	private:
		int numNNLayer;
		int numFltrLayer;
		Layer **fltrLyr;
		LayerDim *nnLayerDim;
		float learnRate;
		float baseLearnRate;
		float dropFactor;
		Network *network;
		value_type *d_nn;
		value_type *qVals;
		value_type rho;
		int miniBatchSize;
		int firstNNLayerUnits;
		int lastNNLayerUnits;
		int totalNNUnits;
		int totalFltrUnits;
		int saveFltrCntr;
		float loss;
		float prevLoss;
		int numSteps;
		int stepSize;
		Info info;
		std::ofstream cnnLog;
		float negSlopeRelu;
	public:
		CNN(Info);
		~CNN();
		void init();
		void initLayers();
		void forwardPropToGetDim();
		void allocateNNMem();
		void forwardProp(value_type*);
		void backwardProp(value_type*);
		int argMaxQVal(int);
		value_type* getQVals();
		void resetNN();
		void printGenAttr();
		void printNNLayerDim();
		void printFltrLayerAttr();
		void testIterate(int);
		void resetQVals();
		void step(value_type*, value_type*);
		int chooseAction(value_type*, int);
		void saveFilterWts();
		value_type* forwardNGetQVal(value_type*);
		int getInputFMSize();
		int getCurWFNum();
		float getCurrentLoss();
		void loadFltrFrmHist();
		void copyFltrToHist();
		void learn(value_type*, value_type*, int);
		float getPrevLoss();
		void resize(int, value_type**);
		int getNumSteps();
		float getLR();
		void printFilterInfo(int, std::string);
		void printAllFltrLyr();
		void printAllNNLyr();
		void testFunctionality();
		void printAllFltrLyrGrad();
};

#endif