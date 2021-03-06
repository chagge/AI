//interface.cu
#include "interface.h"
#include "info.h"
#include "util.h"
#include <string>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <cmath>


Interface::Interface(Info info) {
	dataPath = info.dataPath;
	pathFifoIn = info.pathFifoIn;
	pathFifoOut = info.pathFifoOut;
	numAction = info.numAction;
	maxNumFrame = info.maxNumFrame;
	ind2Act = info.ind2Act;
	act2Ind = info.act2Ind;
	resetButton = info.resetButton;
	numFrmReset = info.numFrmReset;
	numFrmStack = info.numFrmStack;
	cropH = info.cropH;
	cropW = info.cropW;
	cropL = info.cropL;
	cropT = info.cropT;
	cropHV = info.cropHV;
	cropWV = info.cropWV;
	for(int i = 0; i < cropH; ++i) {
		for(int j = 0; j < cropW; ++j) {
			lastFrmGrayInfo.push_back(0);
		}
	}

	frmInfo = dataPath + "/frameInfo";
	EpInfo = dataPath + "/epInfo";

	curFrmNum = 0;
	curAct = UNDEF;
	curIsTerm = 0;
	curEpNum = 0;
	curRew = UNDEF;
	curEpNetRew = 0;
	netGameRew = 0;
	curEpNumFrame = 0;
	delim = ":,";
	lastFrmInfo = "";
	curFrmScreen = "";
	lastEpEndFrmNum = 0;
	minEpFrmGap = 50;
}
Interface::~Interface() {
	dataPath.clear();
	frmInfo.clear();
	EpInfo.clear();
	pathFifoOut.clear();
	pathFifoIn.clear();
	lastFrmInfo.clear();
	delim.clear();
	curFrmScreen.clear();
	ind2Act.clear();
	act2Ind.clear();
	lastFrmGrayInfo.clear();
}

int Interface::openPipe() {
	std::string cmdIn = "cat > " + pathFifoIn;
	std::string cmdOut = "cat " + pathFifoOut;
	if(!(infoIn = popen(cmdIn.c_str(), "w")) || !(infoOut = popen(cmdOut.c_str(), "r"))){
		return FAIL;
	}
	setbuf(infoIn, NULL);
	setbuf(infoOut, NULL);
	return SUCCESS;
}

int Interface::initInPipe() {
	std::string x = "1,0,0,1\n";
	unsigned int numWrite =  fwrite(x.c_str(), sizeof(char), x.length(), infoIn);
	if(numWrite != x.length())
		return FAIL;
	return SUCCESS;
}

void Interface::initOutPipe() {
	char *garbage = inputString(infoOut,10);
	free(garbage);
	garbage = NULL;
	//garbage string
}

int Interface::resetInPipe() {
	std::string x = (toString(resetButton) + ",18\n");
	unsigned int numWrite =  fwrite(x.c_str(), sizeof(char), x.length(), infoIn);
	char *garbage = inputString(infoOut,10);	//garbage string
	free(garbage);
	garbage = NULL;
	if(numWrite != x.length())
		return FAIL;
	return SUCCESS;
}

void Interface::closePipes() {
	pclose(infoIn);
	pclose(infoOut);
}

int Interface::writeInPipe(std::string x) {
	curAct = act2Ind[atoi(x.c_str())];
	x += ",18\n";
	unsigned int numWrite =  fwrite(x.c_str(), sizeof(char), x.length(), infoIn);
	if(numWrite != x.length())
		return FAIL;
	return SUCCESS;
}

void Interface::readFromPipe() {
	lastFrmInfo.clear();
	char *str = inputString(infoOut,10);
	lastFrmInfo.assign(str);
	free(str);
	str = NULL;
	decodeInfo();
}

void Interface::finalizePipe() {
	while(1) {
		std::string x = (toString(18) + ",18\n");
		fwrite(x.c_str(), sizeof(char), x.length(), infoIn);
		char *str = inputString(infoOut,10);
		if(str[0] == 'D') {
			free(str);
			break;
		}
		free(str);
	}
	closePipes();
}

void Interface::decodeInfo() {
	std::vector<int> v;
	for(int i = lastFrmInfo.length()-1; i >= 0; --i) {
		if(lastFrmInfo[i] == ':' || lastFrmInfo[i] == ',') {
			v.push_back(i);
		}
		if(v.size() == 3)
			break;
	}
	curFrmScreen = lastFrmInfo.substr(0, v[2]+1);
	curIsTerm = atoi(lastFrmInfo.substr(v[2]+1, v[1]-v[2]-1).c_str());
	curRew = atoi(lastFrmInfo.substr(v[1]+1, v[0]-v[1]-1).c_str());
	curFrmNum += 1;
	curEpNumFrame +=1;
	curEpNetRew += curRew;
	netGameRew += curRew;
}

int Interface::resetVals(int toSave) {
	if(curIsTerm && (curFrmNum-lastEpEndFrmNum)>minEpFrmGap) {
		if(toSave)
			saveEpisodeInfo();
		curEpNumFrame = 0;
		curEpNetRew = 0;
		curEpNum += 1;
		for(int i = 0; i < numFrmReset; ++i)
			resetInPipe();
		curIsTerm = 0;
		lastEpEndFrmNum = curFrmNum;
		return 1;
	}
	return 0;
}

int Interface::isTerminal() {
	return curIsTerm;
}

int Interface::getCurFrmCnt() {
	return curFrmNum;
}

int Interface::getCurRew() {
	return curRew;
}

int Interface::getCurAct() {
	return curAct;
}

bool Interface::isToEnd() {
	if(curFrmNum >= maxNumFrame - numFrmStack)
		return true;
	return false;
}

int Interface::getCurFrameNum() {
	return curFrmNum;
}

void Interface::preProFrmString() {
	double yRatio = (1.0*(ALE_HEIGHT))/(1.0*cropHV);
	double xRatio = (1.0*(ALE_WIDTH))/(1.0*cropWV);
	for(int i = 0; i < cropHV; ++i) {
		for(int j = 0; j < cropWV; ++j) {
			int firstX = (int)(std::floor(j*xRatio));
			int lastX = (int)(std::floor((j+1)*xRatio));
			int firstY = (int)(std::floor(i*yRatio));
			int lastY = (int)(std::floor((i+1)*yRatio));
			unsigned int resColor = 0.0;
			for(int x = firstX; x <= lastX; ++x) {
				double xRatioInResPixel = 1.0;
				if(x == firstX)
					xRatioInResPixel = x + 1.0 - j*xRatio;
				else if(x == lastX)
					xRatioInResPixel = xRatio*(j+1)-x;

				for(int y = firstY; y <= lastY; ++y) {
					double yRatioInResPixel = 1.0;
					if(y == firstY)
						yRatioInResPixel = y + 1.0 - i*yRatio;
					else if(y == lastY)
						yRatioInResPixel = yRatio*(i+1)-y;
					int grayscale = ntsc2gray(curFrmScreen[(y)*(2*ALE_WIDTH) + 2*(x)], curFrmScreen[(y)*(2*ALE_WIDTH) + 2*(x)+1]);
					resColor += (xRatioInResPixel/xRatio)*(yRatioInResPixel/yRatio)*grayscale;
				}
			}
			//myFrmFile << resColor << " ";
			if(i >= cropT && i < (cropT + cropH) && j >= cropL && j < (cropL + cropW))
				lastFrmGrayInfo[j-cropL+(i-cropT)*cropW] = resColor;
		}
		//myFrmFile << std::endl;
	}

}

std::vector<int> Interface::getGrayScrn(){
	return lastFrmGrayInfo;
}

void Interface::saveFrameInfo() {
	//no need to save frame info
	//std::string path = frmInfo + toString(curFrmNum);
	//std::ofstream myFrmFile(path.c_str());
	//myFrmFile << curFrmScreen << std::endl;
	preProFrmString();
	//myFrmFile << ind2Act[curAct] << std::endl;
	//myFrmFile << curRew << std::endl;
	//myFrmFile << curIsTerm << std::endl;
	//myFrmFile << curFrmNum << std::endl;
	//myFrmFile << curEpNum << std::endl;
	//myFrmFile.close();
}

void Interface::saveEpisodeInfo() {
	std::string path = EpInfo + toString(curEpNum);
	std::ofstream myFrmFile(path.c_str());
	myFrmFile << curEpNum << std::endl;
	myFrmFile << curEpNetRew << std::endl;
	myFrmFile << curEpNumFrame << std::endl;
	myFrmFile.close();
}

void Interface::test() {
	std::cout << "Testing Interface Class START!..." << std::endl;
	std::cout << "TData Path: " << dataPath << std::endl;
	std::cout << "Fifo IN Path: " << pathFifoIn << std::endl;
	std::cout << "Fifo OUT Path: " << pathFifoOut << std::endl;

	std::cout << "Max Number of Frames: " << maxNumFrame << " Max Number of Actions: " << numAction << std::endl;
	for(int i = 0; i < numAction; ++i) {
		std::cout << "Index: " << i << " Action: " << ind2Act[i] << std::endl;
		std::cout << "Action: " << ind2Act[i] << " Index: " << act2Ind[ind2Act[i]] << std::endl;
	}

	std::cout << "Reset Button: " << resetButton << std::endl;
	std::cout << "Num frame stack: " << numFrmStack << std::endl;
	std::cout << "crop Height: " << cropH << std::endl;
	std::cout << "crop Width: " << cropW << std::endl;
	std::cout << "crop Left: " << cropL << std::endl;
	std::cout << "crop Top: " << cropT << std::endl;

	std::cout << "GEN frame info Path: " << frmInfo << std::endl;
	std::cout << "GEN Episode info Path: " << EpInfo << std::endl;
	std::cout << "Opening and Initializing pipe: " << std::endl;
	openPipe();
	initInPipe();
	initOutPipe();
	std::cout << "Writing Random actions to InPipe and reading from OutPipe with data storage..." << std::endl;
	while(!isToEnd()) {
		resetVals(0);
		int x = rand()%numAction;
		writeInPipe(toString(ind2Act[x]));
		readFromPipe();
		saveFrameInfo();
		saveEpisodeInfo();
	}
	finalizePipe();
	std::cout << "Testing Interface Class END!..." << std::endl;
}

int Interface::getCurEpNum() {
	return curEpNum;
}