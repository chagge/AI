caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
  solver_.reset(caffe::GetSolver<float>(solver_param));
  net_ = solver_->net();

   q_values_blob_ = net_->blob_by_name("q_values");

  // Initialize dummy input data with 0
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);

frames_input_layer_ =
      caffe::MemoryDataLayer<float>(
          net_->layer_by_name("frames_input_layer"));



target_input_layer_ =
      caffe::MemoryDataLayer<float>(
          net_->layer_by_name("target_input_layer"));
filter_input_layer_ =
      caffe::MemoryDataLayer<float>(
          net_->layer_by_name("filter_input_layer"));