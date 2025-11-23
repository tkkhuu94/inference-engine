#ifndef INFERENCE_INFERENCE_ENGINE_H_
#define INFERENCE_INFERENCE_ENGINE_H_

#include <vector>

#include "absl/status/statusor.h"
#include "inference/detection.h"
#include "inference/inference_params.h"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

namespace inference {

class InferenceEngine {
public:
  ~InferenceEngine() = default;

  static absl::StatusOr<std::unique_ptr<InferenceEngine>>
  Create(const InferenceParams &params);

  absl::StatusOr<cv::Mat> LetterBox(const cv::Mat &source, int target_w,
                                    int target_h) const;

  absl::StatusOr<std::vector<Detection>>
  RunInference(const cv::Mat &source) const;

private:
  InferenceEngine(const InferenceParams &params);

  InferenceParams params_;
  std::unique_ptr<cv::dnn::Net> net_;
};

} // namespace inference

#endif
