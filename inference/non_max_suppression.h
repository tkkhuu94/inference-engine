#ifndef INFERENCE_NON_MAX_SUPPRESSION_H_
#define INFERENCE_NON_MAX_SUPPRESSION_H_

#include "inference/detection.h"
#include "opencv2/core/types.hpp"

namespace inference {

class NonMaxSuppression {
public:
  static float IoU(const cv::Rect &bbox1, const cv::Rect &bbox2);

  static std::vector<Detection>
  Apply(const std::vector<Detection> &raw_detections, float iou_threshold);
};

} // namespace inference

#endif
