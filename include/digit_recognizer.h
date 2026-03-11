#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>

class DigitRecognizer {
public:
    explicit DigitRecognizer(const std::string& modelPath = "assets/models/mnist.onnx");

    int recognizeDigit(const cv::Mat& digit);
    bool isLoaded() const;

private:
    cv::dnn::Net net_;
    bool loaded_ = false;
};
