#include "inference/non_max_suppression.h"

namespace inference {

float NonMaxSuppression::IoU(const cv::Rect &bbox1, const cv::Rect &bbox2) {
  const auto box1_x1 = bbox1.x;
  const auto box1_x2 = bbox1.x + bbox1.width;
  const auto box1_y1 = bbox1.y;
  const auto box1_y2 = bbox1.y + bbox1.height;
  const auto box1_area = bbox1.width * bbox1.height;

  const auto box2_x1 = bbox2.x;
  const auto box2_x2 = bbox2.x + bbox2.width;
  const auto box2_y1 = bbox2.y;
  const auto box2_y2 = bbox2.y + bbox2.height;
  const auto box2_area = bbox2.width * bbox2.height;

  const auto inter_x1 = std::max(box1_x1, box2_x1);
  const auto inter_x2 = std::min(box1_x2, box2_x2);
  const auto inter_y1 = std::max(box1_y1, box2_y1);
  const auto inter_y2 = std::min(box1_y2, box2_y2);

  const auto inter_width = std::max(0, inter_x2 - inter_x1);
  const auto inter_height = std::max(0, inter_y2 - inter_y1);
  const auto inter_area = inter_width * inter_height;

  if (inter_area == 0) {
    return 0;
  }

  const auto union_area = box1_area + box2_area - inter_area;
  return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

std::vector<Detection>
NonMaxSuppression::Apply(const std::vector<Detection> &raw_detections,
                         float iou_threshold) {
  std::vector<bool> suppressed(raw_detections.size(), false);

  auto sorted_detections = raw_detections;
  std::sort(sorted_detections.begin(), sorted_detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  for (size_t i = 0; i < sorted_detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
      const auto iou =
          IoU(sorted_detections[i].bbox, sorted_detections[j].bbox);

      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  std::vector<Detection> result;
  for (size_t i = 0; i < suppressed.size(); ++i) {
    if (!suppressed[i]) {
      result.emplace_back(sorted_detections[i]);
    }
  }

  return result;
}

} // namespace inference
