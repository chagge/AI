//info.h
#ifndef __INFO_H__
#define __INFO_H__

#include <string>
#include <map>

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

class Info {
	public:
		//interface
		bool isAle;
		int numAction;
		std::map<int, int> ind2Act;
		std::map<int, int> act2Ind;
		int cropH, cropW, cropL, cropT, cropHV, cropWV;
		std::string dataPath;
		int maxNumFrame;
		int numFrmStack;
		//if ale true
		bool isDispScrn;
		std::string romPath;
		//if ale false
		std::string pathFifoIn;
		std::string pathFifoOut;
		int resetButton;
		int numFrmReset;


		//dqn
		bool toTest;
		std::string loadModel;
		//declared above

		//ql
		int maxHistoryLen;
		int miniBatchSize;
		float epsilonDecay;
		std::string solverPath;
		std::string argv0;
		bool toTrain;
		int testAfterEveryNumEp;
		int memThreshold;
		float baseEpsilon;
		int numLearnSteps;
		float futDiscount;
		int targetUpdateFreq;

		//general

		Info();
		~Info();
};

#endif