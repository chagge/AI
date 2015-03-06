//interface.h
#ifndef __INTERFACE_H__
#define __INTERFACE_H__ 1

#include <string>
#include <map>
#include <cstdio>
#include "util.h"
#include "info.h"
#include <vector>
#include <ale_interface.hpp>

class Interface {
	private:
		int maxNumFrame;
		int curFrmNum;
		int curAct;
		int curIsTerm;
		int curEpNum;
		int curRew;
		int curEpNetRew;
		int netGameRew;
		int curEpNumFrame;
		int numAction;
		int lastEpEndFrmNum;
		int minEpFrmGap;
		int numFrmStack;
		int cropHV;
		int cropWV;
		int cropH;
		int cropW;
		int cropL;
		int cropT;
		InputFrames lastFrmGrayInfo;
		std::map<int, int> ind2Act;
		std::map<int, int> act2Ind;
		std::string dataPath;
		std::string EpInfo;
		bool isAle;
		//info.isAle true
		ALEInterface *ale;
		//info.isAle false
		int resetButton;
		int numFrmReset;
		std::string delim;
		std::string curFrmScreen;
		std::string pathFifoIn;
		std::string pathFifoOut;
		std::string lastFrmInfo;
		FILE *infoIn;
		FILE *infoOut;
	public:
		Interface(Info);
		~Interface();
		void init();
		int act(int);
		int resetVals(int);
		int isTerminal();
		bool isToEnd();
		void decodeInfo();
		int getCurFrameNum();
		void saveEpisodeInfo();
		void finalize();
		InputFrames getGrayScrn();
		int getCurEpNum();
		//info.isAle true
		void aleGetScreen();
		void aleAct(int action);
		//info.isAle false
		int openPipe();
		int initInPipe();
		int resetInPipe();
		void initOutPipe();
		void closePipes();
		int writeInPipe(std::string);
		void readFromPipe();
		void finalizePipe();
		void preProFrmString();
		//void test();
};

#endif