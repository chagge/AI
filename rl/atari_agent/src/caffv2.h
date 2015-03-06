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
#include "util.h"
#include "info.h"

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
    Info info;
  public:
    CAFF(Info);
    ~CAFF();
    void Initialize(std::string solver_param);
    void InputDataIntoLayers(const FramesLayerInputData&, const TargetLayerInputData&, const FilterLayerInputData&) ;
    int chooseAction(FramesLayerInputData&, int);
    float *forwardNGetQVal(FramesLayerInputData&);
    void learn(FramesLayerInputData&, TargetLayerInputData&, FilterLayerInputData&, int);
    void loadModel(std::string);
    void copy(CAFF*);
};

#endif
