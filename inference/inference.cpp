#include "absl/log/log.h"
#include "inference/inference_engine.h"
#include "inference_engine.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

void ShowImage(const std::string &image_name, const cv::Mat &image) {
  cv::namedWindow(image_name, cv::WINDOW_NORMAL);
  cv::imshow(image_name, image);
  cv::waitKey(0);
}

int main(int argc, char **argv) {
  // LOG(INFO) << cv::getBuildInformation();

  inference::InferenceParams params{.model_path = "/workspace/yolo11n.onnx",
                                    .input_image_width = 640,
                                    .input_image_height = 640,
                                    .padding_value = cv::Scalar(114, 114, 114)};

  auto engine = inference::InferenceEngine::Create(params);
  if (!engine.ok()) {
    LOG(ERROR) << engine.status();
  }

  cv::Mat source_image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 255));
  // ShowImage("Red", source_image);

  auto detections = (*engine)->RunInference(source_image);
  if (!detections.ok()) {
    LOG(ERROR) << detections.status();
    return 1;
  }

  LOG(INFO) << "Succeeded";

  return 0;
}
