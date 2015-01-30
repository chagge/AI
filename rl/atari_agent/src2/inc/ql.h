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
	public:
		QL(Info);
		~QL();
		void train();
		void initPipes();
		void getAMiniBatch();
		void saveHistory();
		int chooseAction();
		void initSeq();
		void takeAction();
		void learnWts();
		void test();
		void saveGrayScrn();
		int getArgMaxQ();
		void initCNN();
};

#endif