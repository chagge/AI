//ql.cpp
#include "ql.h"
#include "info.h"
#include <cstdlib>
#include "interface.h"
#include <algorithm>
#include <fstream>
#include <iostream>

QL::QL(Info x) {
	info = x;
	numAction = info.numAction;
	ind2Act = info.ind2Act;
	numFrmStack = info.numFrmStack;
	maxHistoryLen = info.maxHistoryLen;
	miniBatchSize = info.miniBatchSize;

	interface = new Interface(info);
	//interface->test();
	epsilonDecay = maxHistoryLen;
	miniBatch = new int[miniBatchSize];
	virtNumTransSaved = 0;
}

QL::~QL() {
	delete[] miniBatch;
	dExp.clear();
	delete interface;
	ind2Act.clear();
	grayScrnHist.clear();
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
	grayScrnHist.push_back(interface->getGrayScrn());
}

void QL::remFrntGrayScrn() {
	grayScrnHist.pop_front();
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

void QL::learnWts() {
	for(int i = 0; i < miniBatchSize; ++i) {
		//set output yj and backpropagate
	}
}

void QL::saveHistory() {
	//history saving cufrmcnt - 1, currew, curact, curisterm, curfrmcnt
	int cnt = interface->getCurFrmCnt();
	History history = {cnt-1, interface->getCurRew(), interface->getCurAct(), interface->isTerminal(), cnt};
	dExp.push_back(history);
	if(dExp.size() >= (unsigned int)maxHistoryLen) {
		dExp.pop_front();
		remFrntGrayScrn();
	}
	virtNumTransSaved++;
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
		return ind2Act[rand()%numAction];	//this has to be replaced with predicted action // argmaxQ
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
		qlLog << cnt - 1 << ", " << interface->getCurRew() << ", " << interface->getCurAct() << ", " << interface->isTerminal() << ", " << cnt << std::endl;
		getAMiniBatch();
		qlLog << "Got miniBatch: " << std::endl;
		for(int i = 0; i < miniBatchSize; ++i) {
			qlLog << miniBatch[i] << ", ";
		}
		qlLog << std::endl;
		learnWts();
	}
	interface->finalizePipe();
	qlLog << "QL Testing END!..." << std::endl;
	qlLog.close();
}