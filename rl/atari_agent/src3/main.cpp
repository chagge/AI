//main.cu
#include "main.h"
#include "info.h"
#include "ql.h"
#include "interface.h"
#include <iostream>
#include <time.h>
//#include "cnn3.h"

int main(int argc, char **argv) {
	srand(time(NULL));
	Info info;
	info.parseArg(argc, argv);
	info.decodeStuff();
	//info.test();
	//std::cout << "------------------------------\n-----------------------------" << std::endl;
	//Interface interface(info);
	//interface.test();

	QL ql(info);
	ql.run();
	//CNN cnn(info);
	//cnn.testFunctionality();
	return 0;
}


//./ale -game_controller fifo_named -run_length_encoding false -disable_color_averaging true ../roms/breakout.bin 
