#ifndef INFERENCE_IMAGE_INFO_H_
#define INFERENCE_IMAGE_INFO_H_

#include "opencv2/opencv.hpp"

namespace inference {

struct ImageInfo {
  float scale;
  int h_padding;
  int w_padding;
  cv::Mat raw_image;
};

} // namespace inference

#endif
