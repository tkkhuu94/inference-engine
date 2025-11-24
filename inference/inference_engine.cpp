#include <cmath>
#include <filesystem>

#include "absl/log/log.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"

#include "inference/inference_engine.h"
#include "inference/non_max_suppression.h"

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

    ptr->net_->enableFusion(false);

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
InferenceEngine::RunInference(const cv::Mat &source) {
  auto letterboxed_image =
      LetterBox(source, params_.input_image_width, params_.input_image_height);
  if (!letterboxed_image.ok()) {
    return letterboxed_image.status();
  }

  auto network_output = Forward(*letterboxed_image);
  if (!network_output.ok()) {
    return network_output.status();
  }

  auto reshaped_output = ParseNetworkOutput(*network_output);
  if (!reshaped_output.ok()) {
    return reshaped_output.status();
  }

  auto detections = ExtractDetections(*reshaped_output);
  if (!detections.ok()) {
    return detections.status();
  }

  auto suppressed_detections =
      NonMaxSuppression::Apply(*detections, params_.iou_threshold);

  return UnscaleDetections(suppressed_detections, source);
}

absl::StatusOr<std::vector<cv::Mat>>
InferenceEngine::Forward(const cv::Mat &image) {

  // Create Blob:
  // - Resize: Already done by Letterbox, but blobFromImage ensures it.
  // - Scale: 1/255.0 (Normalize pixel values 0-255 to 0.0-1.0)
  // - SwapRB: true (OpenCV is BGR, YOLO needs RGB)
  // - Crop: false
  const double scale_factor = 1.0 / 255.0;
  cv::Mat blob = cv::dnn::blobFromImage(
      image, scale_factor,
      cv::Size(params_.input_image_height, params_.input_image_width),
      /*mean*/ cv::Scalar(), /*swapRB*/ true, /*crop*/ false);

  LOG(INFO) << "Blob Size: " << blob.size;

  std::vector<cv::Mat> outs;
  try {
    net_->setInput(blob);
    net_->forward(outs, net_->getUnconnectedOutLayersNames());
  } catch (const cv::Exception &e) {
    return absl::InternalError(e.what());
  }
  return outs;
}

absl::StatusOr<cv::Mat> InferenceEngine::ParseNetworkOutput(
    const std::vector<cv::Mat> &network_output) {
  if (network_output.empty()) {
    return absl::InvalidArgumentError("network output is empty");
  }

  // YOLOv8/v11 Output Shape: [Batch=1, Channels=84, Anchors=8400]
  // We will reshape and return a matrix of shape [8400, 84], such that
  // each row is a detection
  auto output_tensor = network_output.front().reshape(0, 84);

  return output_tensor.t();
}

absl::StatusOr<std::vector<Detection>>
InferenceEngine::ExtractDetections(const cv::Mat &output_tensor) {

  std::vector<Detection> detections;

  for (int i = 0; i < output_tensor.rows; ++i) {
    // Get the row data in pointer, equivalent
    // to detections.row(i) but faster
    const float *row_ptr = output_tensor.ptr<const float>(i);

    float max_confidence_score = 0;
    int class_id = -1;
    for (int id = 4; id < 84; ++id) {
      if (row_ptr[id] > max_confidence_score) {
        max_confidence_score = row_ptr[id];
        class_id = id - 4;
      }
    }

    if (max_confidence_score < params_.confidence_threshold) {
      continue;
    }

    float cx = row_ptr[0];
    float cy = row_ptr[1];
    float w = row_ptr[2];
    float h = row_ptr[3];

    // Convert Center-XYWH to TopLeft-XYWH for OpenCV
    int left = int(cx - w / 2);
    int top = int(cy - h / 2);
    int width = int(w);
    int height = int(h);

    detections.emplace_back(
        Detection{.class_id = class_id,
                  .confidence = max_confidence_score,
                  .bbox = cv::Rect(left, top, width, height)});
  }

  return detections;
}

std::vector<Detection> InferenceEngine::UnscaleDetections(
    const std::vector<Detection> &scaled_detections,
    const cv::Mat &original_image) {
  std::vector<Detection> unscaled_detections;

  const float scale_w = static_cast<float>(params_.input_image_width) /
                        static_cast<float>(original_image.cols);
  const float scale_h = static_cast<float>(params_.input_image_height) /
                        static_cast<float>(original_image.rows);
  const float scale = std::min(scale_w, scale_h);
  LOG(INFO) << "Scale: " << scale;
  LOG(INFO) << "Width: " << original_image.cols;
  LOG(INFO) << "Height: " << original_image.rows;



  int pad_left = (params_.input_image_width - original_image.cols * scale) / 2;
  int pad_top = (params_.input_image_height - original_image.rows * scale) / 2;

  for (const auto &det : scaled_detections) {
    // The Math: x_original = (x_letterboxed - padding) / scale
    float x = (det.bbox.x - pad_left) / scale;
    float y = (det.bbox.y - pad_top) / scale;
    float w = det.bbox.width / scale;
    float h = det.bbox.height / scale;

    // Sanity Clip (Ensure box stays inside the original image)
    // This prevents negative coordinates or boxes spilling off the edge
    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    w = std::min(w, (float)original_image.cols - x);
    h = std::min(h, (float)original_image.rows - y);

    unscaled_detections.emplace_back(
        Detection{.class_id = det.class_id,
                  .confidence = det.confidence,
                  .bbox = cv::Rect((int)x, (int)y, (int)w, (int)h)});
  }

  return unscaled_detections;
}

} // namespace inference
