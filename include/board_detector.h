#pragma once

#include <opencv2/core.hpp>

#include <vector>

class BoardDetector {
public:
    bool detectBoard(const cv::Mat& image, cv::Mat& warpedBoard);

private:
    bool detectSudokuBoardCorners(const cv::Mat& preprocessedImage,
                                  std::vector<cv::Point>& boardCorners,
                                  cv::Mat* debugVisualization = nullptr) const;
    bool findLargestQuadrilateral(const std::vector<std::vector<cv::Point>>& contours,
                                  std::vector<cv::Point>& quadrilateral) const;
    std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point>& corners) const;
};
