#ifndef INFERENCE_DETECTION_H_
#define INFERENCE_DETECTION_H_

namespace inference {

struct BoundingBox {
  int cx;
  int cy;
  int w;
  int h;
};

struct Detection {
  int class_id;
  float confidence;
  BoundingBox bbox;
};

} // namespace inference

#endif
