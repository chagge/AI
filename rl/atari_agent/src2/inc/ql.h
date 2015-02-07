//ql.h
#ifndef __QL_H__
#define __QL_H__ 1

#include "util.h"
#include "info.h"
#include "interface.h"
#include "cnn3.h"
#include <map>
#include <deque>
#include <vector>
#include <string>

class QL {
	private:
		Interface *interface;
		Info info;
		int numAction;
		int numFrmStack;
		std::map<int, int> ind2Act;
		std::deque<History> dExp;	//transitions
		std::vector<int> *grayScrnHist;
		int curLastHistInd;
		double epsilon;
		int epsilonDecay;
		int maxHistoryLen;
		int virtNumTransSaved;
		int miniBatchSize;
		int *miniBatch;
		int numTimeLearnt;
		CNN *cnn;
		int memThreshold;
		int saveWtTimePer;
		float gammaQ;
		std::string qlLogFile;
		float *inputToCNN;
		int cnnInputSize;
		int fMapSize;
		bool isRandom;
		int numInpSaved;
	public:
		QL(Info x);
		~QL();
		void test();
		void setInputToCNN(int lst, int imgInd);
		int getArgMaxQ();
		int chooseAction(bool toTrain);
		double repeatLastAction(int toAct, int x, bool);
		void saveHistory(History history);
		void prepareTarget(float *targ, float *qVals);
		void learnWts();
		double playAnEpisode(bool toTrain);
		void run();
		void getAMiniBatch();
		void printInfo();
		void init();
		void printHistory(History history);
		void finalize();
		void printQVals(float *qVals);
		void printInputToCNN();
		bool gameOver();
		int episodeOver();
		double playAction(int x);
		void saveGrayScrn();
		double initSeq();
		void resetInputToCNN();
		void printInfile(std::ofstream&, float*, int);
};

#endif
