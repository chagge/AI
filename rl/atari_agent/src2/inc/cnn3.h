//cnn3.h
#ifndef __CNN3_H__
#define __CNN3_H__

#include "util.h"
#include "layer.h"
#include "network.h"
#include <string>

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
		int saveFltrCntr;
		std::string dataPath;
	public:
		CNN(std::string, float, float, std::string);
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
		void printFltrLayer(int);
		void printAllFltrLayer();
		void printFltrLayerGrad(int);
		void printAllFltrLayerGrad();
		void testForwardAndBackward();
		void testIterate(int);
		void resetQVals();
		void step(value_type*, value_type*);
		int chooseAction(value_type*, int);
		void saveFilterWts();
		value_type* forwardNGetQVal(value_type*);
		int getInputFMSize();
};

#endif
