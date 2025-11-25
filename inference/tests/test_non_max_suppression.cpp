#include "inference/non_max_suppression.h"
#include "opencv2/core/types.hpp"
#include "gtest/gtest.h"

namespace inference {
namespace {
class NonMaxSuppressionTest : public ::testing::Test {};

TEST_F(NonMaxSuppressionTest, IoUTest) {
  cv::Rect bbox1(0, 0, 10, 10);
  cv::Rect bbox2(5, 5, 10, 10);
  cv::Rect bbox3(15, 15, 10, 10);

  EXPECT_NEAR(NonMaxSuppression::IoU(bbox1, bbox2), 0.1429, 0.001);
  EXPECT_NEAR(NonMaxSuppression::IoU(bbox2, bbox1), 0.1429, 0.001);
  EXPECT_EQ(NonMaxSuppression::IoU(bbox1, bbox1), 1.0);
  EXPECT_EQ(NonMaxSuppression::IoU(bbox1, bbox3), 0);
  EXPECT_EQ(NonMaxSuppression::IoU(bbox2, bbox3), 0);
}

TEST_F(NonMaxSuppressionTest, ApplyTest) {

  Detection a{
      .class_id = 1, .confidence = 0.90f, .bbox = cv::Rect(0, 0, 10, 10)};

  Detection b{
      .class_id = 1, .confidence = 0.80f, .bbox = cv::Rect(1, 1, 10, 10)};

  Detection c{
      .class_id = 1, .confidence = 0.75f, .bbox = cv::Rect(20, 20, 30, 30)};

  Detection d{
      .class_id = 1, .confidence = 0.95f, .bbox = cv::Rect(21, 21, 31, 31)};

  std::vector<Detection> inputs = {a, b, c, d};

  auto result = NonMaxSuppression::Apply(inputs, 0.5f);
  EXPECT_EQ(result.size(), 2);
};

} // namespace
} // namespace inference
