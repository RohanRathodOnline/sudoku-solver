#pragma once

#include <opencv2/core.hpp>

#include <vector>

class CellExtractor {
public:
    std::vector<cv::Mat> extractCells(const cv::Mat& board) const;
};
