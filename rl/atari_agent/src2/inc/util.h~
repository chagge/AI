//util.h
#ifndef __UTIL_H__
#define __UTIL_H__ 1

#define UNDEF -3
#define FAIL 0
#define SUCCESS 1

#include <string>
#include <sstream>
#include <cstdio>


#define ALE_HEIGHT 210
#define ALE_WIDTH 160

struct History{
	int fiJ, reward, act, isTerm, fiJN;
};

inline int toInt(std::string s) {int i;std::stringstream(s)>>i;return i;}
inline std::string toString(int i) {std::string s;std::stringstream ss;ss<<i;ss>>s;return s;}

char *inputString(FILE*, size_t);

double hex2int(char, char);
int ntsc2gray(char, char);

#endif