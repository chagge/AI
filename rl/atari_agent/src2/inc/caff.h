//caff.h
#ifndef __CAFF_H__
#define __CAFF_H__

#include <caffe/caffe.hpp>
#include <string>
#include <algorithm>
#include <iostream>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <array>
#include <cassert>

constexpr auto kCroppedFrameSize = 84;
constexpr auto kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
constexpr auto kInputFrameCount = 4;
constexpr auto kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;
constexpr auto kMinibatchSize = 32;
constexpr auto kMinibatchDataSize = kInputDataSize * kMinibatchSize;
constexpr auto kOutputCount = 3;


using FramesLayerInputData = std::array<float, kMinibatchDataSize>;
using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;


class CAFF {
  private:
    using SolverSp = std::shared_ptr<caffe::Solver<float>>;
    using NetSp = boost::shared_ptr<caffe::Net<float>>;
    using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
    using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;

    SolverSp solver_;
    NetSp net_;
    BlobSp q_values_blob_;
    MemoryDataLayerSp frames_input_layer_;
    MemoryDataLayerSp target_input_layer_;
    MemoryDataLayerSp filter_input_layer_;
    TargetLayerInputData dummy_input_data_;
    float *qVals;
  public:
    CAFF();
    ~CAFF();
    void Initialize(std::string solver_param);
    void InputDataIntoLayers(const FramesLayerInputData&, const TargetLayerInputData&, const FilterLayerInputData&) ;
    int chooseAction(float*, int);
    float *forwardNGetQVal(float*);
    void learn(float*, float*, float*, int);
};

#endif
