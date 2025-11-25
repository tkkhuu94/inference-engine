#ifndef INFERENCE_DETECTION_H_
#define INFERENCE_DETECTION_H_

#include "opencv2/core.hpp"

namespace inference {

struct Detection {
  int class_id;
  float confidence;
  cv::Rect bbox;
};

} // namespace inference

#endif
