//ql.cpp
#include "ql.h"
#include "info.h"
#include <cstdlib>
#include "interface.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include "cnn3.h"

QL::QL(Info x) {
	info = x;
	numAction = info.numAction;
	ind2Act = info.ind2Act;
	numFrmStack = info.numFrmStack;
	maxHistoryLen = info.maxHistoryLen;
	miniBatchSize = info.miniBatchSize;
	gammaQ = info.gammaQ;

	interface = new Interface(info);
	cnn = new CNN(info.nnConfig, info.lr, info.gamma, info.dataPath);
	//interface->test();
	epsilonDecay = maxHistoryLen;
	miniBatch = new int[miniBatchSize];
	virtNumTransSaved = 0;
	grayScrnHist = new std::vector<int>[maxHistoryLen];
	curLastHistInd = 0;
	numTimeLearnt = 0;
	memThreshold = 100;
	saveWtTimePer = 1000;
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
	delete cnn;
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

void QL::initCNN() {
	cnn->init();
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

void QL::learnWts() {
	
	if(virtNumTransSaved < memThreshold)
		return;
	//PREPARE INPUT TO CNN FOR FORWARD PROP
	int fMapSize = cnn->getInputFMSize();
	int cnnInputSize = fMapSize * miniBatchSize * numFrmStack;
	float *inputToCNN = new float[cnnInputSize];
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
	for(int i = 0; i < miniBatchSize; ++i) {
		//set output yj and backpropagate
		int j = 0;
		int cnt = dExp[miniBatch[i]].fiJN;
		cnt = (maxHistoryLen + cnt - 3)%maxHistoryLen;
		while(j < numFrmStack) {
			for(int k = 0; k < fMapSize; ++k) {
				inputToCNN[i*fMapSize*numFrmStack + j*fMapSize + k] = (1.0*grayScrnHist[cnt][k]);
			}
			j++;
			cnt = (cnt + 1)%maxHistoryLen;
		}
	}
	//PREPARE INPUT TO CNN FOR FORWARD PROP COMPLETES HERE
	// fiJ, reward, act, isTerm, fiJN;
	float *qVals = cnn->forwardNGetQVal(inputToCNN);

	//PRINT QVALS IN A FILE TO CHECK IF THEY ARE CONVERGING OR DIVERGING
	std::ofstream qVFile("qValsFile.txt");
	for(int i = 0; i < miniBatchSize*numAction; ++i) {
		qVFile << qVals[i] << " ";
	}
	qVFile << std::endl;
	qVFile.close();
	//PRINT COMPLETES HERE

	//PREPARE TARGET VALUES
	float *targ = new float[miniBatchSize*numAction];
	memset(targ, 0, miniBatchSize*numAction*sizeof(float));
	//target will be zero for those actions which are not performed
	//since we dont know how well would they have done
	for(int i = 0; i < miniBatchSize; ++i) {
		if(!dExp[miniBatch[i]].isTerm) {
			targ[i*numAction + dExp[miniBatch[i]].act] = dExp[miniBatch[i]].reward + gammaQ*qVals[i*numAction + dExp[miniBatch[i]].act];
		} else {
			targ[i*numAction + dExp[miniBatch[i]].act] = dExp[miniBatch[i]].reward;
		}
	}
	//printHostVector(miniBatchSize*numAction, targ);
	//printHostVector(miniBatchSize*numAction, qVals);
	//PREPARE TARGET VALUES COMPLETES HERE

	//PREPARE INPUT TO CNN AGAIN
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
	for(int i = 0; i < miniBatchSize; ++i) {
		//set output yj and backpropagate
		int j = 0;
		int cnt = dExp[miniBatch[i]].fiJ;
		cnt = (maxHistoryLen + cnt - 3)%maxHistoryLen;
		while(j < numFrmStack) {
			for(int k = 0; k < fMapSize; ++k) {
				inputToCNN[i*fMapSize*numFrmStack + j*fMapSize + k] = (1.0*grayScrnHist[cnt][k]);
			}
			j++;
			cnt = (cnt + 1)%maxHistoryLen;
		}
	}
	//PREPARE INPUT TO CNN AGAIN ENDS HERE

	//TAKE A STEP
	cnn->step(inputToCNN, targ);
	
	numTimeLearnt++;
	if(numTimeLearnt%saveWtTimePer==0)
		cnn->saveFilterWts();
	
	delete[] targ;
	delete[] inputToCNN;
	
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
	//PREPARATION OF INPUT TO CNN
	int fMapSize = cnn->getInputFMSize();
	int cnnInputSize = fMapSize * miniBatchSize * numFrmStack;
	float *inputToCNN = new float[cnnInputSize];
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
	int i = 0;
	int cnt = curLastHistInd;
	int ImgInd = 0;
	while(i < numFrmStack) {
		for(int j = 0; j < fMapSize; ++j) {
			inputToCNN[ImgInd*fMapSize*numFrmStack+i*fMapSize+j] = (1.0*grayScrnHist[cnt][j])/255.0 -0.5;
		}
		i++;
		cnt = (maxHistoryLen+cnt-1)%maxHistoryLen;
	}
	//ONLY ONE INPUT HENCE ImgInd need not be used any further
	//PREPARATION OF INPUT TO CNN COMPLETES HERE

	int maxQInd = cnn->chooseAction(inputToCNN, numAction);
	delete[] inputToCNN;
	return maxQInd;
}

int QL::chooseAction() {
	if(interface->getCurFrmCnt() > epsilonDecay)
		epsilon = 0.1;
	else
		epsilon = 1 - (1.0*interface->getCurFrmCnt())/(1.0*epsilonDecay);

	double rn = (1.0*rand())/(1.0*RAND_MAX);
	if(rn < epsilon) {
		std::cout << "Action: " << ind2Act[rand()%numAction] << " RANDOM" << std::endl;
		return ind2Act[rand()%numAction];
	}
	else {
		int amQ = getArgMaxQ();
		std::cout << "Action: " << ind2Act[amQ] << " GREEDY" << std::endl;
		return ind2Act[amQ];
	}
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
	initCNN();
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
		qlLog << "EP No. " << interface->getCurEpNum() << " WF No. " << cnn->getCurWFNum() << std::endl;
		getAMiniBatch();
		learnWts();
		qlLog << "Loss: " << cnn->getCurrentLoss() << std::endl;
	}
	interface->finalizePipe();
	qlLog << "QL Testing END!..." << std::endl;
	qlLog.close();
}
