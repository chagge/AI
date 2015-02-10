//util.cu
#include "util.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>

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

char *inputString(FILE* fp, size_t size){
//The size is extended by the input with the value of the provisional
    char *str;
    int ch;
    size_t len = 0;
    str = (char*)realloc(NULL, sizeof(char)*size);//size is start size
    if(!str)return str;
    while(EOF!=(ch=fgetc(fp)) && ch != '\n'){
        str[len++]=ch;
        if(len==size){
            str = (char*)realloc(str, sizeof(char)*(size+=16));
            if(!str)return str;
        }
    }
    str[len++]='\0';
    return (char*)realloc(str, sizeof(char)*len);
}

double hex2int(char c1, char c2) {
    int x = 0;
    if(c1 >= 'A' && c1 <= 'F') {
        x += 16*(c1 - 'A' + 10);
    } else {
        x += 16*(c1-'0');
    }
    if(c2 >= 'A' && c2 <= 'F') {
        x += (c2 - 'A' + 10);
    } else {
        x += (c2-'0');
    }
    return x;
}

int ntsc2gray(char c0, char c1) {
	int val = hex2int(c0, c1);
	int rgb = ntsc2rgb[val];
  	int r = rgb >> 16;
	int g = (rgb >> 8) & 0xFF;
    int b = rgb & 0xFF;
    return int(r*0.21 + g*0.72 + b*0.07);
}

double rand_normal(double mean, double stddev) {
	static double n2 = 0.0;
	static int n2_cached = 0;
	if (!n2_cached) {
		double x, y, r;
		do {
			x = 2.0*rand()/RAND_MAX - 1;
			y = 2.0*rand()/RAND_MAX - 1;

			r = x*x + y*y;
		} while (r == 0.0 || r > 1.0);
		{
			double d = sqrt(-2.0*log(r)/r);
			double n1 = x*d;
			n2 = y*d;
			double result = n1*stddev + mean;
			n2_cached = 1;
			return result;
		}
	} else {
		n2_cached = 0;
		return n2*stddev + mean;
	}
}

void printDeviceVector(int size, value_type *d_vec) {
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpyDTH(vec, d_vec, size*sizeof(value_type)));
    for (int i = 0; i < size; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
    delete[] vec;
}

void printHostVector(int size, value_type *h_vec) {
    for (int i = 0; i < size; i++)
    {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

void printDeviceVectorInFile(int size, value_type *d_vec, std::ofstream& myFile) {
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpyDTH(vec, d_vec, size*sizeof(value_type)));
    for (int i = 0; i < size; i++)
    {
        myFile << vec[i] << " ";
    }
    myFile << std::endl;
    delete[] vec;
}

void printHostVectorInFile(int size, value_type *h_vec, std::ofstream& myFile, std::string ending = " ") {
    for (int i = 0; i < size; i++)
    {
        myFile << h_vec[i] << ending;
    }
    myFile << std::endl;
}