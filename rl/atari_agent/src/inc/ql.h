//ql.h
#ifndef __QL_H__
#define __QL_H__

#include "util.h"
#include "info.h"
#include "interface.h"
#include <map>

class QL {
	private:
		Interface *interface;
		//CNN cnn;
		Info info;
		int numAction;
		int numFrmStack;
		std::map<int, int> ind2Act;
		History *dExp;	//transitions
		double epsilon;
		int epsilonDecay;
		int maxHistoryLen;
		int numTransSaved;
		int virtNumTransSaved;
		int miniBatchSize;
		int *miniBatch;
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
};

#endif