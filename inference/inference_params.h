#ifndef INFERENCE_INFERENCE_PARAMS_H_
#define INFERENCE_INFERENCE_PARAMS_H_

#include "opencv2/core/types.hpp"

namespace inference {

struct InferenceParams {
  std::string model_path;
  int input_image_width;
  int input_image_height;
  cv::Scalar padding_value;
};

} // namespace inference

#endif
