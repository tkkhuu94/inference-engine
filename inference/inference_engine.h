#ifndef INFERENCE_INFERENCE_ENGINE_H_
#define INFERENCE_INFERENCE_ENGINE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

#include "inference/detection.h"
#include "inference/inference_params.h"

namespace inference {

class InferenceEngine {
public:
  static absl::StatusOr<std::unique_ptr<InferenceEngine>>
  Create(const InferenceParams &params);

  ~InferenceEngine() = default;

  absl::StatusOr<cv::Mat> LetterBox(const cv::Mat &source, int target_w,
                                    int target_h) const;

  absl::StatusOr<std::vector<Detection>> RunInference(const cv::Mat &source);

private:
  InferenceEngine(const InferenceParams &params);

  absl::StatusOr<std::vector<cv::Mat>> Forward(const cv::Mat &image);

  absl::StatusOr<cv::Mat>
  ParseNetworkOutput(const std::vector<cv::Mat> &network_output);

  absl::StatusOr<std::vector<Detection>>
  ExtractDetections(const cv::Mat &output_tensor);

  std::vector<Detection>
  UnscaleDetections(const std::vector<Detection> &scaled_detections,
                    const cv::Mat &original_image);

  InferenceParams params_;
  std::unique_ptr<cv::dnn::Net> net_;
};

} // namespace inference

#endif
