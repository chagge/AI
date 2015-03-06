//main.cu
#include "main.h"
#include "info.h"
#include "ql.h"
#include "interface.h"
#include <iostream>
#include <time.h>

int main(int argc, char **argv) {
	srand(time(NULL));
	Info info;
	QL ql(info);
	ql.run();
	return 0;
}
