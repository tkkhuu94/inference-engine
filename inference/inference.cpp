#include <random>

#include "absl/log/log.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "inference/detection.h"
#include "inference/inference_engine.h"

void SaveImage(const std::string &path, const cv::Mat &image) {
  cv::imwrite(path, image);
}

cv::Scalar GetColor(int class_id) {
  std::srand(class_id);
  int r = std::rand() % 256;
  int g = std::rand() % 256;
  int b = std::rand() % 256;
  return cv::Scalar(b, g, r);
}

void DrawDetections(cv::Mat &img,
                    const std::vector<inference::Detection> &detections,
                    const std::vector<std::string> &class_names) {

  for (const auto &det : detections) {
    cv::Scalar color = GetColor(det.class_id);

    // 1. Draw the Bounding Box
    cv::rectangle(img, det.bbox, color, 3); // Thickness = 3

    // 2. Create the Label Text
    // If class_names provided, use string, else use ID number
    const std::string label = class_names[det.class_id];

    std::string score_string =
        std::to_string((int)(det.confidence * 100)) + "%";
    std::string full_label = label + " " + score_string;

    // 3. Draw Label Background (for readability)
    // Calculate text size
    int base_line;
    cv::Size label_size = cv::getTextSize(full_label, cv::FONT_HERSHEY_SIMPLEX,
                                          0.6, 2, &base_line);

    // Ensure label doesn't go off the top of the image
    int top = std::max(det.bbox.y, label_size.height);

    // Draw filled rectangle behind text
    cv::rectangle(img, cv::Point(det.bbox.x, top - label_size.height),
                  cv::Point(det.bbox.x + label_size.width, top + base_line),
                  color, cv::FILLED);

    // 4. Draw Text (White)
    cv::putText(img, full_label, cv::Point(det.bbox.x, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
  }
}

int main(int argc, char **argv) {
  const std::vector<std::string> kCocoClassNames = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  inference::InferenceParams params{.model_path = "/workspace/yolo11n.onnx",
                                    .input_image_width = 640,
                                    .input_image_height = 640,
                                    .padding_value = cv::Scalar(114, 114, 114),
                                    .confidence_threshold = 0.5,
                                    .iou_threshold = 0.5};

  auto engine = inference::InferenceEngine::Create(params);
  if (!engine.ok()) {
    LOG(ERROR) << engine.status();
  }

  cv::Mat source_image = cv::imread("/workspace/zidane.jpg", cv::IMREAD_COLOR);

  auto detections = (*engine)->RunInference(source_image);
  if (!detections.ok()) {
    LOG(ERROR) << detections.status();
    return 1;
  }

  LOG(INFO) << "Detected " << detections->size() << " objects";

  DrawDetections(source_image, *detections, kCocoClassNames);
  SaveImage("/workspace/detected.jpg", source_image);

  return 0;
}
