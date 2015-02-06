//main.cu
#include "main.h"
#include "info.h"
#include "ql.h"
#include "interface.h"
#include <iostream>

int main(int argc, char **argv) {
	Info info;
	info.parseArg(argc, argv);
	info.decodeStuff();
	//info.test();
	//std::cout << "------------------------------\n-----------------------------" << std::endl;
	//Interface interface(info);
	//interface.test();

	QL ql(info);
	ql.run();
	return 0;
}
