#include <cmath>
#include <filesystem>

#include "absl/log/log.h"
#include "inference/inference_engine.h"
#include "inference_engine.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"

namespace inference {

absl::StatusOr<std::unique_ptr<InferenceEngine>>
InferenceEngine::Create(const InferenceParams &params) {
  if (params.model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty");
  }

  if (!std::filesystem::exists(params.model_path)) {
    return absl::NotFoundError(
        absl::StrFormat("Cannot find model path %s", params.model_path));
  }

  std::unique_ptr<InferenceEngine> ptr(new InferenceEngine(params));
  if (ptr == nullptr) {
    return absl::InternalError("Failed to create the InferenceEngine object");
  }

  try {
    ptr->net_ = std::make_unique<cv::dnn::Net>();
    *(ptr->net_) = cv::dnn::readNetFromONNX(params.model_path);

    auto cuda_devices = cv::cuda::getCudaEnabledDeviceCount();

    auto backend = (cuda_devices > 0) ? cv::dnn::DNN_BACKEND_CUDA
                                      : cv::dnn::DNN_BACKEND_OPENCV;
    auto device =
        (cuda_devices > 0) ? cv::dnn::DNN_TARGET_CUDA : cv::dnn::DNN_TARGET_CPU;

    ptr->net_->setPreferableBackend(backend);
    ptr->net_->setPreferableTarget(device);

    std::string device_name = (cuda_devices > 0) ? "cuda" : "cpu";
    std::string backend_name = (cuda_devices > 0) ? "cuda" : "opencv";
    LOG(INFO) << "Set device to " << device_name;
    LOG(INFO) << "Set backend to " << backend_name;

  } catch (const cv::Exception &opencv_exception) {
    return absl::InternalError(opencv_exception.what());
  }

  return ptr;
}

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

absl::StatusOr<std::vector<Detection>>
InferenceEngine::RunInference(const cv::Mat &source) const {
  auto letterboxed_image =
      LetterBox(source, params_.input_image_width, params_.input_image_height);
  if (!letterboxed_image.ok()) {
    return letterboxed_image.status();
  }

  // Create Blob:
  // - Resize: Already done by Letterbox, but blobFromImage ensures it.
  // - Scale: 1/255.0 (Normalize pixel values 0-255 to 0.0-1.0)
  // - SwapRB: true (OpenCV is BGR, YOLO needs RGB)
  // - Crop: false
  const double scale_factor = 1.0 / 255.0;
  cv::Mat blob = cv::dnn::blobFromImage(
      *letterboxed_image, scale_factor,
      cv::Size(params_.input_image_height, params_.input_image_width),
      /*mean*/ cv::Scalar(), /*swapRB*/ true, /*crop*/ false);

  return std::vector<Detection>();
}

} // namespace inference
