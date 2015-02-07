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
	memThreshold = 500;
	saveWtTimePer = 1000;
	qlLogFile = "qlMainLog.txt";
	numInpSaved = 0;
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
	delete[] inputToCNN;

	delete cnn;
}


void QL::test() {
}

void QL::setInputToCNN(int lst, int imgInd) {
	int i = 0, cnt = (maxHistoryLen + lst - (numFrmStack-1))%maxHistoryLen;
	while(i < numFrmStack) {
		for(int j = 0; j < fMapSize; ++j) {
			inputToCNN[imgInd*fMapSize*numFrmStack+i*fMapSize+j] = (1.0*grayScrnHist[cnt][j])/255.0-0.5;
		}
		i++;
		cnt = (cnt+1)%maxHistoryLen;
	}
}

int QL::getArgMaxQ() {
	resetInputToCNN();
	setInputToCNN(curLastHistInd-1, 0);
	int maxQInd = cnn->chooseAction(inputToCNN, numAction);
	return maxQInd;
}

int QL::chooseAction(bool toTrain) {
	if(numTimeLearnt > epsilonDecay)
		epsilon = 0.1;
	else
		epsilon = 1 - (1.0*numTimeLearnt)/(1.0*epsilonDecay);

	double rn = (1.0*rand())/(1.0*RAND_MAX);
	if(rn < epsilon && toTrain) {
		//std::cout << "Action: " << ind2Act[rand()%numAction] << " RANDOM" << std::endl;
		isRandom = true;
		return rand()%numAction;
	} else {
		int amQ = getArgMaxQ();
		isRandom = false;
		//std::cout << "Action: " << ind2Act[amQ] << " GREEDY" << std::endl;
		return amQ;
	}
}

double QL::repeatLastAction(int toAct, int x, bool toTrain) {
	double lastScore = 0;
	for(int i = 0; i < x && !gameOver(); ++i) {
		lastScore += playAction(toAct);
	}
	if(toTrain)
		saveGrayScrn();
	return lastScore;
}

void QL::saveHistory(History history) {
	dExp.push_back(history);
	if(dExp.size() >= (unsigned int)maxHistoryLen) {
		dExp.pop_front();
	}
	virtNumTransSaved++;
}

void QL::prepareTarget(float *targ, float *qVals) {
	//target will be zero for those actions which are not performed
	//since we dont know how well would they have done
	for(int i = 0; i < miniBatchSize; ++i) {
		if(!dExp[miniBatch[i]].isTerm) {
			float maxQV = 0;
			int maxQVI = 0;
			for(int j = 0; j < numAction; ++j) {
				if(qVals[i*numAction + j] > qVals[i*numAction + maxQVI]) {
					maxQVI = j;
				}
			}
			maxQV = qVals[i*numAction + maxQVI];	
			targ[i*numAction + dExp[miniBatch[i]].act] = dExp[miniBatch[i]].reward + gammaQ*maxQV;
		} else {
			targ[i*numAction + dExp[miniBatch[i]].act] = dExp[miniBatch[i]].reward;
		}
	}
}

void QL::printInfile(std::ofstream& myF, float *val, int sz) {
	for(int i = 0; i < sz; ++i) {
		myF << val[i] << " ";
	}
	myF << std::endl;
	myF << std::endl;
}

void QL::learnWts() {

	std::ofstream logFile("mainLog.txt", std::fstream::out | std::fstream::app);
	
	resetInputToCNN();
	for(int i = 0; i < miniBatchSize; ++i) {
		setInputToCNN(dExp[miniBatch[i]].fiJN, i);
	}

	float *qVals = cnn->forwardNGetQVal(inputToCNN);

	logFile << "ITERATION NO.: " << numTimeLearnt << std::endl;
	logFile << "Predicted QVals: " << std::endl;
	printInfile(logFile, qVals, miniBatchSize*numAction);
	//printQVals(qVals);

	float *targ = new float[miniBatchSize*numAction];
	memset(targ, 0, miniBatchSize*numAction*sizeof(float));
	prepareTarget(targ, qVals);

	logFile << "Target was: " << std::endl;
	printInfile(logFile, targ, miniBatchSize*numAction);

	resetInputToCNN();
	for(int i = 0; i < miniBatchSize; ++i) {
		setInputToCNN(dExp[miniBatch[i]].fiJ, i);
	}

	//TAKE A STEP
	int maxLIter = 1;
	cnn->learn(inputToCNN, targ, maxLIter);

	logFile << "New Predicted QVals: " << std::endl;
	printInfile(logFile, cnn->getQVals(), miniBatchSize*numAction);
	logFile.close();

	numTimeLearnt++;
	if(numTimeLearnt%saveWtTimePer==0)
		cnn->saveFilterWts();
	
	delete[] targ;
	
}

double QL::playAnEpisode(bool toTrain) {
	double epScore = 0;
	int ftime = 1;
	while(!gameOver() && !interface->isTerminal()) {
		#ifdef TESTQL
			std::cout << "Epoch Started" << std::endl;
		#endif
		if(ftime == 1) {
			ftime = 0;
			epScore += initSeq();
		}
		int toAct = chooseAction(toTrain);
		double lastScore = repeatLastAction(ind2Act[toAct], numFrmStack, toTrain);
		epScore += lastScore;
		int reward = 0;
		if(toTrain) {
			if(lastScore != 0.0f) {
				reward = 1;
				if(lastScore < 0.0f) {
					reward = -1;
				}
			}
			History history = {(maxHistoryLen+curLastHistInd-2)%maxHistoryLen, 
								reward, 
								toAct, 
								interface->isTerminal(), 
								(maxHistoryLen+curLastHistInd-1)%maxHistoryLen};
			saveHistory(history);
			#ifdef TESTQL
				printHistory(history);
			#endif
			if(dExp.size() > memThreshold) {
				getAMiniBatch();
				learnWts();
			}
		}
		#ifdef TESTQL
			std::cout << "Epoch Ended" << std::endl;
		#endif
	}
	interface->resetVals(1);
	return epScore;
}

void QL::run() {
	#ifdef TESTQL
		printInfo();
	#endif
	init();
	//while(1)
	std::ofstream qlFile(qlLogFile.c_str());
	while(!gameOver()) {
		double score = playAnEpisode(true);
		qlFile << "EP No. " << interface->getCurEpNum() << " and score: " << score << " WF No. " << cnn->getCurWFNum() << std::endl;
		if((interface->getCurEpNum())%10 == 0) {
			score = playAnEpisode(false);
			qlFile << "Test: " << interface->getCurEpNum() << " and score: " << score << std::endl;
		}
	}
	finalize();
}

void QL::getAMiniBatch() {
	for(int i = 0; i < miniBatchSize; ++i) {
		miniBatch[i] = rand()%(dExp.size());
	}
}

void QL::printInfo() {
	std::cout << "QL Testing START!..." << std::endl;
	std::cout << "Number of Actions: " << numAction << std::endl;
	std::cout << "Size of frame stack: " << numFrmStack << std::endl;
	std::cout << "Max History Length: " << maxHistoryLen << std::endl;
	std::cout << "Minibatch size: " << miniBatchSize << std::endl;
	std::cout << "Actions: " << std::endl;
	for(int i = 0; i < numAction; ++i) {
		std::cout << "Action with index: " << i << " is: " << ind2Act[i] << std::endl;
	}
}

void QL::init() {
	//init pipes
	interface->openPipe();
	interface->initInPipe();
	interface->initOutPipe();
	//init cnn
	cnn->init();
	//init input to CNN
	fMapSize = cnn->getInputFMSize();
	cnnInputSize = fMapSize * miniBatchSize * numFrmStack;
	inputToCNN = new float[cnnInputSize];
	resetInputToCNN();
}

void QL::printHistory(History history) {
	std::cout << "prev Hist Ind: " << history.fiJ << std::endl;
	std::cout << "Reward: " << history.reward << std::endl;
	std::cout << "Acted: " << history.act << (isRandom?" RANDOM ":" DECISION ") << std::endl;
	std::cout << "is Terminal: " << history.isTerm << std::endl;
	std::cout << "new Hist Ind: " << history.fiJN << std::endl;
}

void QL::finalize() {
	interface->finalizePipe();
}

void QL::printQVals(float *qVals) {
	//PRINT QVALS IN A FILE TO CHECK IF THEY ARE CONVERGING OR DIVERGING
	std::ofstream qVFile("qValsFile.txt");
	for(int i = 0; i < miniBatchSize*numAction; ++i) {
		qVFile << qVals[i] << " ";
	}
	qVFile << std::endl;
	qVFile.close();
	//PRINT COMPLETES HERE
}

void QL::printInputToCNN() {
	std::string path = info.dataPath + "/inputToCNN" + toString(numInpSaved);
	numInpSaved++;
	std::ofstream myF(path.c_str());
	for(int i = 0; i < miniBatchSize; ++i) {
		for(int j = 0; j < numFrmStack; ++j) {
			for(int k = 0; k < fMapSize; ++k) {
				myF << inputToCNN[i*fMapSize*numFrmStack + j*fMapSize + k] << std::endl;
			}
		}
	}
	myF.close();
}

bool QL::gameOver() {
	return interface->isToEnd();
}

int QL::episodeOver() {
	return interface->isTerminal();
}

double QL::playAction(int x) {
	interface->writeInPipe(toString(x));
	interface->readFromPipe();
	interface->saveFrameInfo();
	return interface->getCurRew();
}

void QL::saveGrayScrn() {
	grayScrnHist[curLastHistInd%maxHistoryLen] = interface->getGrayScrn();
	curLastHistInd = (curLastHistInd+1)%maxHistoryLen;
}

double QL::initSeq() {
	double netScore = 0;
	for(int i = 0; i < numFrmStack && !gameOver(); ++i) {
		netScore += playAction(ind2Act[rand()%numAction]);
		saveGrayScrn();
	}
	return netScore;
}

void QL::resetInputToCNN() {
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
}