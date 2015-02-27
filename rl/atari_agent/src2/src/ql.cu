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
	epsilonDecay = info.epsilonDecay;

	interface = new Interface(info);
	cnn = new CNN(x);
	miniBatch = new int[miniBatchSize];
	grayScrnHist = new std::vector<int>[maxHistoryLen];
	
	virtNumTransSaved = 0;
	lastHistInd = 0;
	numTimeLearnt = 0;

	//garbage vals
	isRandom = false;

	fMapSize = cnn->getInputFMSize();
	cnnInputSize = fMapSize * miniBatchSize * numFrmStack;
	inputToCNN = new float[cnnInputSize];
}

QL::~QL() {
	delete[] miniBatch;
	delete interface;
	ind2Act.clear();
	for(int i = 0; i < maxHistoryLen; ++i) {
		grayScrnHist[i].clear();
	}
	delete[] grayScrnHist;
	delete[] inputToCNN;
	delete cnn;
	dExp.clear();
	if(info.debugQL) {
		qlLog.close();
		qlLog2.close();
	}
}

void QL::run() {
	init();

	if(info.debugQL)
		printParamInfo();

	while(!gameOver()) {

		double score = playAnEpisode(info.toTrain);
		if(info.debugQL) {
			qlLog << "Train: EP No. " << interface->getCurEpNum() << " and score: " << score << " WF No. " << cnn->getCurWFNum() << std::endl;
			qlLog2 << "Train: EP No. " << interface->getCurEpNum() << " and score: " << score << " WF No. " << cnn->getCurWFNum() << std::endl;
			qlLog2 << "Num steps Connet: " << cnn->getNumSteps() << " Learning rate: " << cnn->getLR() << std::endl;
		}
		if((interface->getCurEpNum())%info.testAfterEveryNumEp == 0 && info.toTrain) {
			score = playAnEpisode(false);	//tests in training
			if(info.debugQL) {
				qlLog << "Test: EP No. " << interface->getCurEpNum() << " and score: " << score << std::endl;
				qlLog2 << "Test: EP No. " << interface->getCurEpNum() << " and score: " << score << " WF No. " << cnn->getCurWFNum() << std::endl;
			}
		}
	}

	finalize();
}

void QL::init() {
	//init pipes
	interface->openPipe();
	interface->initInPipe();
	interface->initOutPipe();
	
	//init input to CNN
	resetInputToCNN();

	//open file if info.debugQL is on
	if(info.debugQL) {
		qlLog.open(info.qlLogFile.c_str());
		qlLog2.open(info.qlLogFile2.c_str());
	}
}

bool QL::gameOver() {
	return interface->isToEnd();
}

double QL::playAnEpisode(bool toTrain) {
	double epScore = 0;
	int ftime = 1;
	int frameNum = 0;

	if(info.debugQL)
		qlLog << "Episode No.: " << interface->getCurEpNum() << std::endl;

	while(!gameOver() && !interface->isTerminal()) {

		if(info.debugQL)
			qlLog << "Frame No.: " << frameNum << std::endl;

		if(ftime == 1) {
			ftime = 0;
			epScore += initSeq();
		}

		int toAct = chooseAction(toTrain);
		if(info.debugQL)
			qlLog << "Action chosen." << std::endl;
		double lastScore = repeatLastAction(ind2Act[toAct], numFrmStack, toTrain);
		if(info.debugQL)
			qlLog << "Action repeated." << std::endl;
		epScore += lastScore;
		int reward = 0;
		if(lastScore != 0.0f) {
			reward = 1;
			if(lastScore < 0.0f) {
				reward = -1;
			}
		}

		if(info.debugQL)
			qlLog << "Action Chosen: " << toAct << (isRandom?" (Random)":" (Decisive)") << " got Reward: " << reward << std::endl;

		if(toTrain) {
			History history = {(maxHistoryLen+lastHistInd-2)%maxHistoryLen, 
								reward, 
								toAct, 
								interface->isTerminal(), 
								(maxHistoryLen+lastHistInd-1)%maxHistoryLen};
			saveHistory(history);
			if(info.debugQL) {
				qlLog << "prev Hist Ind: " << history.fiJ << std::endl;
				qlLog << "is Terminal: " << history.isTerm << std::endl;
				qlLog << "new Hist Ind: " << history.fiJN << std::endl;
			}
			if(dExp.size() > info.memThreshold) {
				getAMiniBatch();
				learnWts();
			}
		}
		frameNum++;
	}
	interface->resetVals(1);
	return epScore;
}

double QL::initSeq() {
	double netScore = 0;
	for(int i = 0; i < numFrmStack && !gameOver(); ++i) {
		netScore += playAction(ind2Act[rand()%numAction]);
		saveGrayScrn();
	}
	return netScore;
}

double QL::playAction(int x) {
	interface->writeInPipe(toString(x));
	interface->readFromPipe();
	interface->saveFrameInfo();
	return interface->getCurRew();
}

void QL::saveGrayScrn() {
	grayScrnHist[lastHistInd%maxHistoryLen] = interface->getGrayScrn();
	lastHistInd = (lastHistInd+1)%maxHistoryLen;
}

int QL::chooseAction(bool toTrain) {
	float epsilon;
	if(numTimeLearnt > epsilonDecay || !toTrain)
		epsilon = info.baseEpsilon;
	else
		epsilon = 1 - 0.9*((1.0*numTimeLearnt)/(1.0*epsilonDecay));

	double rn = (1.0*rand())/(1.0*RAND_MAX);
	if(rn < epsilon) {
		isRandom = true;
		return rand()%numAction;
	} else {
		int amQ = getArgMaxQ();
		isRandom = false;
		return amQ;
	}
}

int QL::getArgMaxQ() {
	resetInputToCNN();
	setInputToCNN((maxHistoryLen+lastHistInd-1)%maxHistoryLen, 0);
	int maxQInd = cnn->chooseAction(inputToCNN, numAction);
	return maxQInd;
}

double QL::repeatLastAction(int toAct, int numTimes, bool toTrain) {
	double lastScore = 0;
	for(int i = 0; i < numTimes && !gameOver(); ++i) {
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

void QL::getAMiniBatch() {
	for(int i = 0; i < miniBatchSize; ++i) {
		miniBatch[i] = rand()%(dExp.size());
	}
}

void QL::learnWts() {
	resetInputToCNN();
	for(int i = 0; i < miniBatchSize; ++i) {
		setInputToCNN(dExp[miniBatch[i]].fiJN, i);
	}

	float *qVals = cnn->forwardNGetQVal(inputToCNN);
	if(info.debugQL) {
		qlLog << "Learn iter no.: " << numTimeLearnt << std::endl;
		qlLog << "Predicted QVals: " << std::endl;
		printInfile(qlLog, qVals, miniBatchSize*numAction);
		qlLog << "Action and Rewards in miniBatch: " << std::endl;
		for(int i = 0; i < miniBatchSize; ++i) {
			qlLog << dExp[miniBatch[i]].act << "<==>" << dExp[miniBatch[i]].reward << ", ";
		}
		qlLog << std::endl;
		if(numTimeLearnt < 1) {
			std::ofstream myF("inputToCNNB.txt");
			printHostVectorInFile(cnnInputSize, inputToCNN, myF, "\n");
			myF.close();
		}
	}

	float *targ = new float[miniBatchSize*numAction];
	memset(targ, 0, miniBatchSize*numAction*sizeof(float));
	prepareTarget(targ, qVals);

	if(info.debugQL) {
		qlLog << "Target was: " << std::endl;
		printInfile(qlLog, targ, miniBatchSize*numAction);
	}

	resetInputToCNN();
	for(int i = 0; i < miniBatchSize; ++i) {
		setInputToCNN(dExp[miniBatch[i]].fiJ, i);
	}

	//TAKE A STEP
	int maxLIter = info.numLearnSteps;
	cnn->learn(inputToCNN, targ, maxLIter);

	if(info.debugQL) {
		qlLog << "New predicted QVals: " << std::endl;
		printInfile(qlLog, cnn->forwardNGetQVal(inputToCNN), miniBatchSize*numAction);
		if(numTimeLearnt < 1) {
			std::ofstream myF("inputToCNNA.txt");
			printHostVectorInFile(cnnInputSize, inputToCNN, myF, "\n");
			myF.close();
		}
	}

	numTimeLearnt++;
	if(numTimeLearnt%info.saveWtTimePer==0)
		cnn->saveFilterWts();
	
	delete[] targ;
	
}

void QL::resetInputToCNN() {
	memset(inputToCNN, 0, cnnInputSize*sizeof(float));
}

void QL::setInputToCNN(int lst, int imgInd) {
	int i = 0, cnt = (maxHistoryLen + lst - (numFrmStack-1))%maxHistoryLen;
	while(i < numFrmStack) {
		for(int j = 0; j < fMapSize; ++j) {
			inputToCNN[imgInd*fMapSize*numFrmStack+i*fMapSize+j] = (1.0*grayScrnHist[cnt][j])/255.0;
		}
		i++;
		cnt = (cnt+1)%maxHistoryLen;
	}
}

void QL::printInfile(std::ofstream& myF, float *val, int sz) {
	for(int i = 0; i < sz; ++i) {
		myF << val[i] << ", ";
	}
	myF << std::endl;
	myF << std::endl;
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
			targ[i*numAction + dExp[miniBatch[i]].act] = dExp[miniBatch[i]].reward + info.futDiscount*maxQV;
		} else {
			targ[i*numAction + dExp[miniBatch[i]].act] = dExp[miniBatch[i]].reward;
		}
	}
}

void QL::finalize() {
	interface->finalizePipe();
}

void QL::printParamInfo() {
	qlLog << "Number of Actions: " << numAction << std::endl;
	qlLog << "Size of frame stack: " << numFrmStack << std::endl;
	qlLog << "Max History Length: " << maxHistoryLen << std::endl;
	qlLog << "Epsilon Decay: " << epsilonDecay << std::endl;
	qlLog << "Minibatch size: " << miniBatchSize << std::endl;
	qlLog << "Actions: " << std::endl;
	for(int i = 0; i < numAction; ++i) {
		qlLog << "Action with index: " << i << " is: " << ind2Act[i] << std::endl;
	}
}