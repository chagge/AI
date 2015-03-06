//info.cpp

#include "info.h"
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>

Info::Info() {
	//interface
	isAle = true;
	numAction = 3;
	ind2Act[0] = 11;
	ind2Act[1] = 12;
	ind2Act[2] = 1;
	act2Ind[11] = 0;
	act2Ind[12] = 1;
	act2Ind[1] = 2;
	dataPath = "../data";
	cropH = 84;
	cropW = 84;
	cropHV = 110;
	cropWV = 84;
	cropL = 0;
	cropT = 18;
	maxNumFrame = 50000000;
	numFrmStack = 4;

	if(isAle) {
		isDispScrn = true;
		romPath = "../roms/pong.bin";
	} else {
		pathFifoIn = "";
		pathFifoOut = "";
		numFrmReset = 4;
		resetButton = 45;
	}

	toTest = false;
	loadModel = "model3/model_iter_995000.caffemodel";

	maxHistoryLen = 1000000;
	miniBatchSize = 32;
	epsilonDecay = 1000000;
	solverPath = "../config_files/dqn_solver.prototxt";
	argv0 = "atari";
	toTrain = true;;
	testAfterEveryNumEp = 10;
	memThreshold = 100;
	baseEpsilon = 0.1f;
	numLearnSteps = 1;
	futDiscount = 0.99f;
	targetUpdateFreq = 10000;
}

Info::~Info() {

}