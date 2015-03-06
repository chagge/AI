//ql.h
#ifndef __QL_H__
#define __QL_H__ 1

#include "util.h"
#include "info.h"
#include "interface.h"
//#include "cnn3.h"
#include <map>
#include <deque>
#include <vector>
#include <string>
#include <fstream>
#include "caff.h"

class QL {
	private:
		Interface *interface;
		Info info;
		int numAction;
		int numFrmStack;
		std::map<int, int> ind2Act;
		std::deque<History> dExp;	//transitions
		std::deque<InputFrames> grayScrnHist;
		int lastHistInd;
		int epsilonDecay;
		int maxHistoryLen;
		int virtNumTransSaved;
		int miniBatchSize;
		int *miniBatch;
		int numTimeLearnt;
		FramesLayerInputData inputToCNN;
		int cnnInputSize;
		int fMapSize;
		bool isRandom;
		CAFF *caff;
		CAFF *caff2;
		int targetUpdate;
	public:
		QL(Info x);
		~QL();
		void run();
		void init();
		bool gameOver();
		double playAnEpisode(bool);
		double initSeq();
		int chooseAction(bool);
		int getArgMaxQ();
		double playAction(int);
		void saveGrayScrn();
		double repeatLastAction(int, int, bool);
		void saveHistory(History);
		void getAMiniBatch();
		void learnWts();
		void printInfile(int sz, FramesLayerInputData val , std::ofstream& myF);
		void resetInputToCNN();
		void setInputToCNN(int, int);
		void prepareTarget(TargetLayerInputData&, float*);
		void finalize();
		void printParamInfo();
};

#endif
