#pragma once

#include <opencv2/core.hpp>

class GridCleaner {
public:
    cv::Mat removeGridLines(const cv::Mat& board) const;
};
