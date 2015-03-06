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
		int resetButton;
		int numFrmReset;
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
		std::string delim;
		std::string curFrmScreen;
		std::string dataPath;
		std::string frmInfo;
		std::string EpInfo;
		std::string pathFifoIn;
		std::string pathFifoOut;
		std::string lastFrmInfo;
		std::map<int, int> ind2Act;
		std::map<int, int> act2Ind;
		FILE *infoIn;
		FILE *infoOut;
		ALEInterface *ale;
	public:
		Interface(Info);
		~Interface();
		int openPipe();
		int initInPipe();
		int resetInPipe();
		void initOutPipe();
		void closePipes();
		int writeInPipe(std::string);
		void readFromPipe();
		void finalizePipe();
		void decodeInfo();
		int resetVals(int);
		int isTerminal();
		bool isToEnd();
		int getCurFrameNum();
		void saveFrameInfo();
		void saveEpisodeInfo();
		void test();
		int getCurFrmCnt();
		int getCurRew();
		int getCurAct();
		void preProFrmString();
		InputFrames getGrayScrn();
		int getCurEpNum();
		void aleGetScreen();
		void aleAct(int action);
};

#endif