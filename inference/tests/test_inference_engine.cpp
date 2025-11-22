#include "inference/inference_engine.h"
#include "opencv2/core.hpp"
#include "gtest/gtest.h"

namespace inference {
namespace {
class InferenceEngineTest : public ::testing::Test {
protected:
  InferenceEngineTest()
      : engine_(InferenceParams{.padding_value = cv::Scalar(114, 114, 114)}) {}

  InferenceEngine engine_;
};

TEST_F(InferenceEngineTest, LetterBoxTest) {
  cv::Mat source_image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 255));
  auto letterboxed_result = engine_.LetterBox(source_image, 640, 640);
  ASSERT_TRUE(letterboxed_result.ok());
  ASSERT_EQ(letterboxed_result->rows, 640);
  ASSERT_EQ(letterboxed_result->cols, 640);

  // Verify the top paddings are set to padding value
  for (int c = 0; c < 640; c++) {
    for (int r = 0; r < 140; r++) {
      ASSERT_EQ((*letterboxed_result).at<cv::Vec3b>(r, c)[0], 114);
    }
  }

  // Verify the bottom paddings are set to padding value
  for (int c = 0; c < 640; c++) {
    for (int r = 500; r < 640; r++) {
      ASSERT_EQ((*letterboxed_result).at<cv::Vec3b>(r, c)[0], 114);
    }
  }

  // Verify the scaled image are copied to the center
  for (int c = 0; c < 640; c++) {
    for (int r = 140; r < 500; r++) {
      ASSERT_EQ((*letterboxed_result).at<cv::Vec3b>(r, c)[2], 255);
    }
  }

}

} // namespace
} // namespace inference
