#ifndef INFERENCE_INFERENCE_ENGINE_H_
#define INFERENCE_INFERENCE_ENGINE_H_

#include "absl/status/statusor.h"
#include "inference/inference_params.h"
#include "opencv2/core.hpp"

namespace inference {

class InferenceEngine {
public:
  InferenceEngine(const InferenceParams &params);
  ~InferenceEngine() = default;

  absl::StatusOr<cv::Mat> LetterBox(const cv::Mat &source, int target_w,
                                    int target_h) const;

private:
  InferenceParams params_;
};

} // namespace inference

#endif
