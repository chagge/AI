//info.cu
#include "info.h"
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>

Info::Info() {
	//empty conrtuctor
	lr = 0.001;
	gamma = 0.1;
	gammaQ = 0.5;
}
Info::~Info() {
	aleConfig.clear();
	fifoConfig.clear();
	dataPath.clear();
	pathFifoIn.clear();
	pathFifoOut.clear();
	ind2Act.clear();
	act2Ind.clear();
	nnConfig.clear();
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
		else if(temp == "-n" || temp == "--nn_config")
		{
			if(argc != i)
				nnConfig = argv[i];
			else
			{
				std::cout << "invalid fifo config file: use -h, --help for usage information" << std::endl;
				exit(EXIT_FAILURE);
			}
		}			
		else if(temp == "-l" || temp == "--learn_rate")
		{
			if(argc != i)
				lr = atof(argv[i]);
			else
			{
				std::cout << "invalid learning rate: use -h, --help for usage information" << std::endl;
				exit(EXIT_FAILURE);
			}
		}		
		else if(temp == "-g" || temp == "--gamma")
		{
			if(argc != i)
				gamma = atof(argv[i]);
			else
			{
				std::cout << "invalid gamma: use -h, --help for usage information" << std::endl;
				exit(EXIT_FAILURE);
			}
		}		
		else if(temp == "-gq" || temp == "--gamma_q")
		{
			if(argc != i)
				gammaQ = atof(argv[i]);
			else
			{
				std::cout << "invalid gamma: use -h, --help for usage information" << std::endl;
				exit(EXIT_FAILURE);
			}
		}					
		else if(temp == "-h" || temp == "--help")
		{
			std::cout << "atari: Atari learning agent\n\nUsage:atari [options] [filename...]\nDescription\n\nArguments: \n-h, --help \n\t\t show this help message and exit\n-a, --ale_config <filename> \n\t\t read ale configurations from this file\n-f, --fifo_config <filename> \n\t\t read fifo config from this file\n-l, --learn_rate <lr> \n\t\t cnn learn rate\n-g, --gamma <g> \n\t\t cnn rmsprop gamma\n-gq, --gamma_q <g> \n\t\t Q learner gamma(discount) for future rewards" << std::endl;
			exit(EXIT_SUCCESS);
		}
		i--;
	}

	if(aleConfig.empty() || fifoConfig.empty() || nnConfig.empty())
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
	ale >> cropH;
	ale >> cropW;
	ale >> cropL;
	ale >> cropT;
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
	std::cout << "Num frame stack: " << numFrmStack << std::endl;
	std::cout << "maximum history length: " << maxHistoryLen << std::endl;
	std::cout << "crop Height: " << cropH << std::endl;
	std::cout << "crop Width: " << cropW << std::endl;
	std::cout << "crop Left: " << cropL << std::endl;
	std::cout << "crop Top: " << cropT << std::endl;

	std::cout << "Decoded FIFOCONFIG output: " << std::endl;
	std::cout << "Data Path: " << dataPath << std::endl;
	std::cout << "Fifo IN Path: " << pathFifoIn << std::endl;
	std::cout << "Fifo OUT Path: " << pathFifoOut << std::endl;
	std::cout << "Testing Info Class DONE!..." << std::endl;
}
