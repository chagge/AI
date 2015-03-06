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
#include <ale_interface.hpp>


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
	ale = new ALEInterface(true);
	ale->loadROM("pong.bin");
	/*
	for(int i = 0; i < cropH; ++i) {
		for(int j = 0; j < cropW; ++j) {
			lastFrmGrayInfo[j+i*cropW] = 0;
		}
	}
	*/

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
	//lastFrmGrayInfo.clear();
}

int Interface::openPipe() {
	return SUCCESS;
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
	return SUCCESS;
	std::string x = "1,0,0,1\n";
	unsigned int numWrite =  fwrite(x.c_str(), sizeof(char), x.length(), infoIn);
	if(numWrite != x.length())
		return FAIL;
	return SUCCESS;
}

void Interface::initOutPipe() {
	return;
	char *garbage = inputString(infoOut,10);
	free(garbage);
	garbage = NULL;
	//garbage string
}

int Interface::resetInPipe() {
	return SUCCESS;
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
	return;
	pclose(infoIn);
	pclose(infoOut);
}

int Interface::writeInPipe(std::string x) {
	aleAct(atoi(x.c_str()));
	return SUCCESS;
	curAct = act2Ind[atoi(x.c_str())];
	x += ",18\n";
	unsigned int numWrite =  fwrite(x.c_str(), sizeof(char), x.length(), infoIn);
	if(numWrite != x.length())
		return FAIL;
	return SUCCESS;
}

void Interface::readFromPipe() {
	aleGetScreen();
	return;
	lastFrmInfo.clear();
	char *str = inputString(infoOut,ALE_WIDTH*ALE_HEIGHT*2);
	lastFrmInfo.assign(str);
	free(str);
	str = NULL;
	decodeInfo();
}

void Interface::finalizePipe() {
	return;
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
	return;
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
		ale->reset_game();
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

int PixelToGrayscale(pixel_t pixel) {

int ntsc2rgb[] = {
	0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
	0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
	0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
	0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
	0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
	0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
	0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
	0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
	0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
	0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
	0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
	0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
	0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
	0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
	0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
	0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
	0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
	0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
	0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
	0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
	0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
	0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
	0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
	0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
	0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
	0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
	0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
	0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
	0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
	0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
	0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
	0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
};
	int rgb = ntsc2rgb[pixel];
  	int r = rgb >> 16;
	int g = (rgb >> 8) & 0xFF;
    int b = rgb & 0xFF;
    return int(r*0.21 + g*0.72 + b*0.07);
}

void Interface::preProFrmString() {
	return;
	auto screen = std::make_shared<FrameData>();
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
				(*screen)[j-cropL+(i-cropT)*cropW] = resColor;
		}
		//myFrmFile << std::endl;
	}
	lastFrmGrayInfo[0] = screen;
}

void Interface::aleGetScreen() {
	const auto raw_pixels = ale->getScreen().getArray();

	auto screen = std::make_shared<FrameData>();
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
					int grayscale = PixelToGrayscale(raw_pixels[static_cast<int>(y * ALE_WIDTH + x)]);
					resColor += (xRatioInResPixel/xRatio)*(yRatioInResPixel/yRatio)*grayscale;
				}
			}
			//myFrmFile << resColor << " ";
			if(i >= cropT && i < (cropT + cropH) && j >= cropL && j < (cropL + cropW))
				(*screen)[j-cropL+(i-cropT)*cropW] = resColor;
		}
		//myFrmFile << std::endl;
	}
	lastFrmGrayInfo[0] = screen;
}

void Interface::aleAct(int action) {
	curIsTerm = ale->game_over();
	curRew = ale->act((Action)action);
	curFrmNum += 1;
	curEpNumFrame +=1;
	curEpNetRew += curRew;
	netGameRew += curRew;
}

InputFrames Interface::getGrayScrn(){
	return lastFrmGrayInfo;
}

void Interface::saveFrameInfo() {
	return;
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

	std::cout << "Reset Button:Interface:: " << resetButton << std::endl;
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
