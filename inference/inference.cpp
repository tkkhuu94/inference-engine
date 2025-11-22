#include "absl/log/log.h"
#include "inference/inference_engine.h"
#include "opencv2/imgcodecs.hpp"

void SaveImage(const std::string &path, const cv::Mat &image) {
  cv::imwrite(path, image);
}

int main(int argc, char **argv) {
  inference::InferenceEngine engine(
      inference::InferenceParams{.padding_value = cv::Scalar(114, 114, 114)});

  cv::Mat source_image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 255));
  SaveImage("/home/tri/Downloads/original.jpg", source_image);

  auto letterboxed_image = engine.LetterBox(source_image, 640, 640);
  if (!letterboxed_image.ok()) {
    LOG(ERROR) << letterboxed_image.status();
    return 1;
  }
  SaveImage("/home/tri/Downloads/letterboxed.jpg", *letterboxed_image);

  return 0;
}
