#ifndef INFERENCE_INFERENCE_PARAMS_H_
#define INFERENCE_INFERENCE_PARAMS_H_

#include "opencv2/core.hpp"

namespace inference {

struct InferenceParams {
  cv::Scalar padding_value;
};

} // namespace inference

#endif
