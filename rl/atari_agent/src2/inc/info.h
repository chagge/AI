//info.h
#ifndef __INFO_H__
#define __INFO_H__ 1

#include <string>
#include <map>

class Info{
	public:
		//for ql
		int numAction;
		int numFrmStack;
		int maxHistoryLen;
		int epsilonDecay;
		int miniBatchSize;
		bool debugQL;
		bool toTrain;
		int testAfterEveryNumEp;
		int memThreshold;
		std::string qlLogFile, qlLogFile2;
		float baseEpsilon;
		int numLearnSteps;
		int saveWtTimePer;
		float futDiscount;

		//for ql and interface
		std::map<int, int> ind2Act;
		std::map<int, int> act2Ind;

		//for nn
		bool debugCNN;
		bool isBias;
		std::string cnnLogFile;

		std::string nnConfig;	//input
		std::string aleConfig;	//input
		std::string fifoConfig;	//input
		std::string dataPath;
		std::string pathFifoIn;
		std::string pathFifoOut;
		int maxNumFrame;
		int resetButton;
		int numFrmReset;
		int cropH;
		int cropW;
		int cropL;
		int cropT;
		Info();
		~Info();
		void parseArg(int, char**);
		void decodeStuff();
		void test();
};
#endif
