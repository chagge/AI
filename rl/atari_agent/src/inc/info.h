//info.h
#ifndef __INFO_H__
#define __INFO_H__ 1

#include <string>
#include <map>

class Info{
	public:
		int numAction;
		std::string aleConfig;	//input
		std::string fifoConfig;	//input
		std::string dataPath;
		std::string pathFifoIn;
		std::string pathFifoOut;
		int maxNumFrame;
		std::map<int, int> ind2Act;
		std::map<int, int> act2Ind;
		int resetButton;
		int numFrmReset;
		int numFrmStack;
		int maxHistoryLen;
		int miniBatchSize;
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