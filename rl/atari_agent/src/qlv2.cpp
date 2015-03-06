//ql.cpp
#include "ql.h"
#include "info.h"
#include <cstdlib>
#include "interface.h"
#include <algorithm>
#include <fstream>
#include <iostream>
//#include "cnn3.h"
#include "caff.h"


QL::QL(Info x) {
	info = x;
	numAction = info.numAction;
	ind2Act = info.ind2Act;
	numFrmStack = info.numFrmStack;
	maxHistoryLen = info.maxHistoryLen;
	miniBatchSize = info.miniBatchSize;
	epsilonDecay = info.epsilonDecay;

	interface = new Interface(info);
	miniBatch = new int[miniBatchSize];
	
	virtNumTransSaved = 0;
	lastHistInd = 0;
	numTimeLearnt = 0;
	targetUpdate = 0;

	//garbage vals
	isRandom = false;
	fMapSize = kCroppedFrameDataSize;
	cnnInputSize = fMapSize * miniBatchSize * numFrmStack;

	google::InitGoogleLogging(info.argv0.c_str());
	google::InstallFailureSignalHandler();
	google::LogToStderr();
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	caff = new CAFF(x);
	caff->Initialize(info.solverPath);
	caff2 = new CAFF(x);
	caff2->Initialize(info.solverPath);
	caff2 = caff;
}

QL::~QL() {
	delete[] miniBatch;
	delete interface;
	ind2Act.clear();
	dExp.clear();
}

void QL::run() {
	init();

	while(!gameOver()) {

		double score = playAnEpisode(info.toTrain);
		if((interface->getCurEpNum())%info.testAfterEveryNumEp == 0 && info.toTrain) {
			score = playAnEpisode(false);	//tests in training
		}
		std::cout << "Frame: " << interface->getCurFrameNum() << " Ep: " <<  interface->getCurEpNum() << " score: " << score << std::endl;
	}
	finalize();
}

void QL::init() {
	interface->init();
	resetInputToCNN();
}

bool QL::gameOver() {
	return interface->isToEnd();
}

double QL::playAnEpisode(bool toTrain) {
	double epScore = 0;
	int ftime = 1;

	while(!gameOver() && !interface->isTerminal()) {
		if(ftime == 1) {
			ftime = 0;
			epScore += initSeq();
		}
		int toAct = chooseAction(toTrain);
		double lastScore = repeatLastAction(ind2Act[toAct], numFrmStack, toTrain);
		epScore += lastScore;
		int reward = 0;
		if(lastScore != 0.0f) {
			reward = 1;
			if(lastScore < 0.0f) {
				reward = -1;
			}
		}

		if(toTrain) {
			History history = {(maxHistoryLen+lastHistInd-2)%maxHistoryLen, 
								reward, 
								toAct, 
								interface->isTerminal(), 
								(maxHistoryLen+lastHistInd-1)%maxHistoryLen};
			saveHistory(history);
			if(dExp.size() > info.memThreshold) {
				getAMiniBatch();
				learnWts();
			}
		}
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
	return interface->act(x);
}

void QL::saveGrayScrn() {
	grayScrnHist.push_back(interface->getGrayScrn());
	if(grayScrnHist.size() >= maxHistoryLen) {
		grayScrnHist.pop_front();
	}
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
	int maxQInd = caff->chooseAction(inputToCNN, numAction);
	return maxQInd;
}

double QL::repeatLastAction(int toAct, int numTimes, bool toTrain) {
	double lastScore = 0;
	for(int i = 0; i < numTimes && !gameOver(); ++i) {
		lastScore += playAction(toAct);
	}
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
	if(numTimeLearnt < 1) {
		std::ofstream myF("inputToCNN.txt");
		printInfile(cnnInputSize, inputToCNN, myF);
		myF.close();
	}

	float *qVals = caff2->forwardNGetQVal(inputToCNN);

	TargetLayerInputData targ;
	std::fill(targ.begin(), targ.end(), 0.0f);
	FilterLayerInputData filterInp;
	std::fill(filterInp.begin(), filterInp.end(), 0.0f);
	prepareTarget(targ, qVals);

	for(int i = 0; i < miniBatchSize; ++i) {
		filterInp[i*numAction + dExp[miniBatch[i]].act] = 1;
	}

	resetInputToCNN();
	for(int i = 0; i < miniBatchSize; ++i) {
		setInputToCNN(dExp[miniBatch[i]].fiJ, i);
	}

	int maxLIter = info.numLearnSteps;
	caff->learn(inputToCNN, targ, filterInp, maxLIter);

	numTimeLearnt++;
	if(numTimeLearnt - targetUpdate >= info.targetUpdateFreq) {
		targetUpdate = numTimeLearnt;
		caff2 = caff;
	}
}

void QL::resetInputToCNN() {
	std::fill(inputToCNN.begin(), inputToCNN.end(), 0.0f);
}

void QL::setInputToCNN(int lst, int imgInd) {
	int i = 0, cnt = (maxHistoryLen + lst - (numFrmStack-1))%maxHistoryLen;
	while(i < numFrmStack) {
		const auto& frame_data = grayScrnHist[cnt][0];
		std::copy(frame_data->begin(), frame_data->end(), inputToCNN.begin()+imgInd*fMapSize*numFrmStack+i*fMapSize);
		i++;
		cnt = (cnt+1)%maxHistoryLen;
	}
}

void QL::printInfile(int sz, FramesLayerInputData val , std::ofstream& myF) {
	for(int i = 0; i < sz; ++i) {
		myF << val[i] << "\n";
	}
	myF << std::endl;
}

void QL::prepareTarget(TargetLayerInputData& targ, float *qVals) {
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
	interface->finalize();
}

void QL::printParamInfo() {
	std::cout << "Number of Actions: " << numAction << std::endl;
	std::cout << "Size of frame stack: " << numFrmStack << std::endl;
	std::cout << "Max History Length: " << maxHistoryLen << std::endl;
	std::cout << "Epsilon Decay: " << epsilonDecay << std::endl;
	std::cout << "Minibatch size: " << miniBatchSize << std::endl;
	std::cout << "Actions: " << std::endl;
	for(int i = 0; i < numAction; ++i) {
		std::cout << "Action with index: " << i << " is: " << ind2Act[i] << std::endl;
	}
}
