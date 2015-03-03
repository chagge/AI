//util.h
#ifndef __UTIL_H__
#define __UTIL_H__ 1

#define UNDEF -3
#define FAIL 0
#define SUCCESS 1

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>

constexpr auto kRawFrameHeight = 210;
constexpr auto kRawFrameWidth = 160;
constexpr auto kCroppedFrameSize = 84;
constexpr auto kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
constexpr auto kInputFrameCount = 4;
constexpr auto kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;
constexpr auto kMinibatchSize = 32;
constexpr auto kMinibatchDataSize = kInputDataSize * kMinibatchSize;
constexpr auto kOutputCount = 3;

using FrameData = std::array<float, kCroppedFrameDataSize>;
using FrameDataSp = std::shared_ptr<FrameData>;
using InputFrames = std::array<FrameDataSp, 1>;
using Transition = std::tuple<InputFrames, int, float, boost::optional<FrameDataSp>>;

using FramesLayerInputData = std::array<float, kMinibatchDataSize>;
using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;

#define ALE_HEIGHT 210
#define ALE_WIDTH 160

typedef float value_type;

struct History{
	int fiJ, reward, act, isTerm, fiJN;
};

struct LayerDim{
	int x, y, z, w;
};
inline int toInt(std::string s) {int i;std::stringstream(s)>>i;return i;}
inline std::string toString(int i) {std::string s;std::stringstream ss;ss<<i;ss>>s;return s;}

char *inputString(FILE*, size_t);

int hex2int(char, char);
int ntsc2gray(char, char);
double rand_normal(double, double);
void printHostVector(int, value_type*);
void printHostVectorInFile(int, value_type*, std::ofstream&, std::string);

#endif
