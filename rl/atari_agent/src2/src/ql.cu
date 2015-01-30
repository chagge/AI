//ql.cpp
#include "ql.h"
#include "info.h"
#include <cstdlib>
#include "interface.h"
#include <algorithm>
#include <fstream>
#include <iostream>
//#include "cnn3.h"

QL::QL(Info x) {
	info = x;
	numAction = info.numAction;
	ind2Act = info.ind2Act;
	numFrmStack = info.numFrmStack;
	maxHistoryLen = info.maxHistoryLen;
	miniBatchSize = info.miniBatchSize;

	interface = new Interface(info);
	//cnn = new CNN(info);
	//interface->test();
	epsilonDecay = maxHistoryLen;
	miniBatch = new int[miniBatchSize];
	virtNumTransSaved = 0;
	grayScrnHist = new std::vector<int>[maxHistoryLen];
	curLastHistInd = 0;
	numTimeLearnt = 0;
}

QL::~QL() {
	delete[] miniBatch;
	dExp.clear();
	delete interface;
	ind2Act.clear();
	for(int i = 0; i < maxHistoryLen; ++i) {
		grayScrnHist[i].clear();
	}
	delete[] grayScrnHist;
}

void QL::train() {
	int fTime = 1;
	//init pipes
	initPipes();

	while(!interface->isToEnd()) {
		if(interface->resetVals(1) || fTime) {
			fTime = 0;
			initSeq();
		}
		takeAction();
		saveHistory();
		getAMiniBatch();
		learnWts();
	}
	interface->finalizePipe();
}

void QL::initPipes() {
	interface->openPipe();
	interface->initInPipe();
	interface->initOutPipe();
}

void QL::saveGrayScrn() {
	grayScrnHist[interface->getCurFrmCnt()%maxHistoryLen] = interface->getGrayScrn();
	curLastHistInd = interface->getCurFrmCnt()%maxHistoryLen;
}

void QL::initSeq() {
	for(int i = 0; i < numFrmStack; ++i) {
		int x = rand()%numAction;
		interface->writeInPipe(toString(ind2Act[x]));
		interface->readFromPipe();
		interface->saveFrameInfo();
		saveGrayScrn();
	}
}

void QL::takeAction() {
	int toAct = chooseAction();
	interface->writeInPipe(toString(toAct));
	interface->readFromPipe();
	interface->saveFrameInfo();
	saveGrayScrn();
}

void QL::getAMiniBatch() {
	for(int i = 0; i < miniBatchSize; ++i) {
		miniBatch[i] = rand()%(dExp.size());
	}
}

void QL::learnWts() {/*
	numTimeLearnt++;
	if(virtNumTransSaved < 100)
		return;
	int fMapSize = grayScrnHist[1].size();
	int cnnInputSize = fMapSize * miniBatchSize * numFrmStack;
	float *inputToCNN = new float[cnnInputSize];
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
	for(int i = 0; i < miniBatchSize; ++i) {
		//set output yj and backpropagate
		int j = 0;
		int cnt = dExp[miniBatch[i]].fiJN;
		while(j < numFrmStack) {
			for(int k = 0; k < fMapSize; ++k) {
				inputToCNN[i*fMapSize*numFrmStack + j*fMapSize + k] = (1.0*grayScrnHist[cnt][k])/255.0;
			}
			j++;
			cnt = (maxHistoryLen + cnt - 1)%maxHistoryLen;
		}
	}

	cnn->initInputLayer(inputToCNN);
	cnn->forwardProp();
	float *qVals = cnn->getQVals();
	float *err = new float[numAction*miniBatchSize];
	memset(err, 0, numAction*miniBatchSize*sizeof(float));
	
	for(int i = 0; i < miniBatchSize; ++i) {
		std::cout << qVals[i*numAction] << " " << qVals[i*numAction+1] << std::endl;
		err[i*numAction + dExp[miniBatch[i]].act] = -dExp[miniBatch[i]].reward + qVals[i*numAction+dExp[miniBatch[i]].act];
	}
	cnn->initOutputErr(err);
	cnn->backPropagate();
	if(numTimeLearnt%1000==0)
		cnn->saveWeights();
	
	delete[] err;
	delete[] qVals;
	delete[] inputToCNN;
	*/
}

void QL::saveHistory() {
	//history saving cufrmcnt - 1, currew, curact, curisterm, curfrmcnt
	int cnt = interface->getCurFrmCnt();
	History history = {(maxHistoryLen + cnt-1)%maxHistoryLen, interface->getCurRew(), interface->getCurAct(), interface->isTerminal(), (cnt)%maxHistoryLen};
	dExp.push_back(history);
	if(dExp.size() >= (unsigned int)maxHistoryLen) {
		dExp.pop_front();
	}
	virtNumTransSaved++;
}

int QL::getArgMaxQ() {
	return 0;
	/*

	int fMapSize = grayScrnHist[1].size();
	int cnnInputSize = fMapSize * miniBatchSize * numFrmStack;
	float *inputToCNN = new float[cnnInputSize];
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
	int i = 0;
	int cnt = curLastHistInd;
	while(i < numFrmStack) {
		for(int j = 0; j < fMapSize; ++j) {
			inputToCNN[j] = (1.0*grayScrnHist[cnt][j])/255.0;
		}
		i++;
		cnt = (maxHistoryLen+cnt-1)%maxHistoryLen;
	}
	cnn->initInputLayer(inputToCNN);
	cnn->forwardProp();
	float *qVals = cnn->getQVals();
	int maxQInd = 0;
	for(int i = 0; i < numAction; ++i) {
		if(qVals[maxQInd] < qVals[i])
			maxQInd = i;
	}
	delete[] qVals;
	delete[] inputToCNN;
	return maxQInd;
	*/
}

int QL::chooseAction() {
	if(interface->getCurFrmCnt() > epsilonDecay)
		epsilon = 0.1;
	else
		epsilon = 1 - (1.0*interface->getCurFrmCnt())/(1.0*epsilonDecay);

	double rn = (1.0*rand())/(1.0*RAND_MAX);
	if(rn < epsilon)
		return ind2Act[rand()%numAction];
	else
		return ind2Act[getArgMaxQ()];	//this has to be replaced with predicted action // argmaxQ
}

void QL::test() {
	std::ofstream qlLog("ql.txt");

	qlLog << "QL Testing START!..." << std::endl;
	qlLog << "Number of Actions: " << numAction << std::endl;
	qlLog << "Size of frame stack: " << numFrmStack << std::endl;
	qlLog << "Max History Length: " << maxHistoryLen << std::endl;
	qlLog << "Minibatch size: " << miniBatchSize << std::endl;
	qlLog << "Actions: " << std::endl;
	for(int i = 0; i < numAction; ++i) {
		qlLog << "Action with index: " << i << " is: " << ind2Act[i] << std::endl;
	}
	int fTime = 1;
	//init pipes
	initPipes();
	qlLog << "Pipes Initiated!..." << std::endl;
	while(!interface->isToEnd()) {
		if(interface->resetVals(1) || fTime) {
			fTime = 0;
			initSeq();
		}
		takeAction();
		saveHistory();
		qlLog << "Virtual History Number: " << virtNumTransSaved << std::endl;
		qlLog << "History Saved: " << std::endl;
		int cnt = interface->getCurFrmCnt();
		qlLog << (maxHistoryLen + cnt - 1)%maxHistoryLen << ", " << interface->getCurRew() << ", " << interface->getCurAct() << ", " << interface->isTerminal() << ", " << cnt%maxHistoryLen  << std::endl;
		getAMiniBatch();
		learnWts();
	}
	interface->finalizePipe();
	qlLog << "QL Testing END!..." << std::endl;
	qlLog.close();
}
