#include <caffe/caffe.hpp>
#include <string>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <cassert>
#include "caff.h"
#include <cstdlib>

CAFF::~CAFF() {
  delete[] qVals;
}

CAFF::CAFF(Info x) {
  info = x;
}

void CAFF::Initialize(std::string solver_param_) {
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
  solver_.reset(caffe::GetSolver<float>(solver_param));
  net_ = solver_->net();
  frames_input_layer_ =
  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
  net_->layer_by_name("frames_input_layer"));
  target_input_layer_ =
  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
  net_->layer_by_name("target_input_layer"));
  filter_input_layer_ =
  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
  net_->layer_by_name("filter_input_layer"));
  q_values_blob_ = net_->blob_by_name("q_values");
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0); 
  qVals = new float[kOutputCount*kMinibatchSize];
  if(info.toTest) {
    loadModel(info.loadModel);
  }
}

void CAFF::InputDataIntoLayers(
      const FramesLayerInputData& frames_input,
      const TargetLayerInputData& target_input,
      const FilterLayerInputData& filter_input) {
    frames_input_layer_->Reset(const_cast<float*>(frames_input.data()),
                  dummy_input_data_.data(),
                  kMinibatchSize);
    target_input_layer_->Reset(const_cast<float*>(target_input.data()),
                  dummy_input_data_.data(),
                  kMinibatchSize);
    filter_input_layer_->Reset(const_cast<float*>(filter_input.data()),
                  dummy_input_data_.data(),
                  kMinibatchSize);
}

int CAFF::chooseAction(FramesLayerInputData& frame_data, int numAction) {
  assert(numAction == kOutputCount);
  InputDataIntoLayers(frame_data, dummy_input_data_, dummy_input_data_);
  net_->ForwardPrefilled(nullptr);
  int maxIdx = 0;
  float maxQ;

  for(int i = 0; i < numAction; ++i) {
    float q =q_values_blob_->data_at(0, static_cast<int>(i), 0, 0);
    assert(!std::isnan(q));
    if(i == 0)
      maxQ = q;
    else if(q>maxQ) {
      maxQ = q;
      maxIdx = i;
    }
  }
  return maxIdx;
}


float *CAFF::forwardNGetQVal(FramesLayerInputData& frame_data) {
  InputDataIntoLayers(frame_data, dummy_input_data_, dummy_input_data_);
  net_->ForwardPrefilled(nullptr);
  memset(qVals, 0, kOutputCount*kMinibatchSize);
  for(int i = 0; i < kMinibatchSize; ++i) {
    for(int j = 0; j < kOutputCount; ++j) {
      qVals[j+i*kOutputCount] = q_values_blob_->data_at(static_cast<int>(i), static_cast<int>(j), 0, 0);
    }
  }
  return qVals;
}

void CAFF::learn(FramesLayerInputData& frames_input, TargetLayerInputData& target_input, FilterLayerInputData& filter_input, int iter) {
  InputDataIntoLayers(frames_input, target_input, filter_input);
  solver_->Step(iter);
}

void CAFF::loadModel(std::string modelPath) {
  net_->CopyTrainedLayersFrom(modelPath);
}

void CAFF::copy(CAFF *cf) {
  net_ = cf->net_;
  solver_ = cf->solver_;
}