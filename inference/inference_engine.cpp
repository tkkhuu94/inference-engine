#include "inference/inference_engine.h"
#include "inference_engine.h"
#include "opencv2/imgproc.hpp"
#include <cmath>

namespace inference {

InferenceEngine::InferenceEngine(const InferenceParams &params)
    : params_(params) {}

absl::StatusOr<cv::Mat> InferenceEngine::LetterBox(const cv::Mat &source,
                                                   int target_w,
                                                   int target_h) const {
  const float scale_w =
      static_cast<float>(target_w) / static_cast<float>(source.cols);
  const float scale_h =
      static_cast<float>(target_h) / static_cast<float>(source.rows);

  const float scale = std::min(scale_w, scale_h);

  int result_w = static_cast<int>(source.cols * scale);
  int result_h = static_cast<int>(source.rows * scale);

  cv::Mat result(target_h, target_w, CV_8UC3, params_.padding_value);

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(result_w, result_h));

  int top = std::abs(target_h - result_h) / 2;
  int left = std::abs(target_w - result_w) / 2;

  try {
    resized.copyTo(result(cv::Rect(left, top, result_w, result_h)));
  } catch (const cv::Exception &e) {
    return absl::InternalError(e.what());
  }

  return result;
}

} // namespace inference
