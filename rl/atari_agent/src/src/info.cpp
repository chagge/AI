//info.cpp
#include "info.h"
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>

Info::Info() {
	//empty conrtuctor
}
Info::~Info() {
	aleConfig.clear();
	fifoConfig.clear();
	dataPath.clear();
	pathFifoIn.clear();
	pathFifoOut.clear();
}

void Info::parseArg(int argc, char **argv) {
	int i = argc;
	while(i > 1)
	{
		std::string temp(argv[i - 1]);
		if(temp == "-a" || temp == "--ale_config")
		{
			if(argc != i)
				aleConfig = argv[i];
			else
			{
				std::cout << "invalid ale config file: use -h, --help for usage information" << std::endl;
				exit(EXIT_FAILURE);
			}
		}
		else if(temp == "-f" || temp == "--fifo_config")
		{
			if(argc != i)
				fifoConfig = argv[i];
			else
			{
				std::cout << "invalid fifo config file: use -h, --help for usage information" << std::endl;
				exit(EXIT_FAILURE);
			}
		}				
		else if(temp == "-h" || temp == "--help")
		{
			std::cout << "atari: Atari learning agent\n\nUsage:atari [options] [filename...]\nDescription\n\nArguments: \n-h, --help \n\t\t show this help message and exit\n-a, --ale_config <filename> \n\t\t read ale configurations from this file\n-f, --fifo_config <filename> \n\t\t read fifo config from this file" << std::endl;
			exit(EXIT_SUCCESS);
		}
		i--;
	}

	if(aleConfig.empty() || fifoConfig.empty())
	{
		std::cout << "input files not provided: use -h, --help for usage information" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Info::decodeStuff() {
	std::ifstream ale(aleConfig.c_str());
	ale >> maxNumFrame;
	ale >> numAction;
	ale >> resetButton;
	ale >> numFrmReset;
	ale >> numFrmStack;
	ale >> maxHistoryLen;
	int x, i=0;
	while(ale >> x) {
		ind2Act[i]=x;
		act2Ind[x]=i;
		i++;
	}
	ale.close();

	miniBatchSize = 32;

	std::ifstream fifo(fifoConfig.c_str());
	fifo >> dataPath;
	fifo >> pathFifoIn;
	fifo >> pathFifoOut;
	fifo.close();
}

void Info::test() {
	std::cout << "Testing Info Class START!..." << std::endl;
	std::cout << "ALECONFIG FILE: " << aleConfig << std::endl;
	std::cout << "FIFOCONFIG FILE: " << fifoConfig << std::endl;
	std::cout << "Decoded ALECONFIG output: " << std::endl;
	std::cout << "Max Number of Frames: " << maxNumFrame << " Max Number of Actions: " << numAction << std::endl;
	for(int i = 0; i < numAction; ++i) {
		std::cout << "Index: " << i << " Action: " << ind2Act[i] << std::endl;
		std::cout << "Action: " << ind2Act[i] << " Index: " << act2Ind[ind2Act[i]] << std::endl;
	}
	std::cout << "Reset Button: " << resetButton << std::endl;

	std::cout << "Decoded FIFOCONFIG output: " << std::endl;
	std::cout << "Data Path: " << dataPath << std::endl;
	std::cout << "Fifo IN Path: " << pathFifoIn << std::endl;
	std::cout << "Fifo OUT Path: " << pathFifoOut << std::endl;
	std::cout << "Testing Info Class DONE!..." << std::endl;
}