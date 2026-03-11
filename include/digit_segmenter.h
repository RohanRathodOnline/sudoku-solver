#pragma once

#include <opencv2/core.hpp>

class DigitSegmenter {
public:
    bool extractDigit(const cv::Mat& cell, cv::Mat& digit) const;

private:
    cv::Mat normalizeTo28x28(const cv::Mat& componentMask) const;
};
