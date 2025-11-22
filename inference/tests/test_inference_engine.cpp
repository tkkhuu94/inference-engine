#include "inference/inference_engine.h"
#include "opencv2/core.hpp"
#include "gtest/gtest.h"

namespace inference {
namespace {
class InferenceEngineTest : public ::testing::Test {
protected:
  InferenceEngineTest()
      : engine_(InferenceParams{.padding_value = cv::Scalar(114, 114, 114)}) {}

  static bool IsRegionSolidColor(const cv::Mat &full_image, cv::Rect region,
                                 cv::Scalar expected_color);

  InferenceEngine engine_;
};

bool InferenceEngineTest::IsRegionSolidColor(const cv::Mat &full_image,
                                             cv::Rect region,
                                             cv::Scalar expected_color) {
  // Extract the ROI
  // This is O(1)
  cv::Mat roi = full_image(region);

  // Calculate absolute difference between ROI and the scalar
  cv::Mat diff;
  cv::absdiff(roi, expected_color, diff);

  // Sum all the differences across all channels
  cv::Scalar total_diff = cv::sum(diff);

  // If sum is 0 in all channels, the region is identical to the scalar
  return (total_diff[0] == 0.0 && total_diff[1] == 0.0 && total_diff[2] == 0.0);
}

TEST_F(InferenceEngineTest, LetterBoxTest) {
  cv::Mat source_image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 255));
  auto letterboxed_result = engine_.LetterBox(source_image, 640, 640);
  ASSERT_TRUE(letterboxed_result.ok());
  ASSERT_EQ(letterboxed_result->rows, 640);
  ASSERT_EQ(letterboxed_result->cols, 640);

  cv::Rect scaled_region(0, 140, 640, 360);
  cv::Rect top_pad_rect(0, 0, 640, 140);
  cv::Rect bottom_pad_rect(0, 500, 640, 140); // 640 - 500 = 140 height

  EXPECT_TRUE(InferenceEngineTest::IsRegionSolidColor(
      *letterboxed_result, top_pad_rect, cv::Scalar(114, 114, 114)))
      << "Top padding contained non-grey pixels!";

  EXPECT_TRUE(InferenceEngineTest::IsRegionSolidColor(
      *letterboxed_result, bottom_pad_rect, cv::Scalar(114, 114, 114)))
      << "Bottom padding contained non-grey pixels!";

  EXPECT_TRUE(InferenceEngineTest::IsRegionSolidColor(
      *letterboxed_result, scaled_region, cv::Scalar(0, 0, 255)))
      << "Scaled image contained non-red pixels!";
}

} // namespace
} // namespace inference
