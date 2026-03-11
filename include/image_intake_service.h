#pragma once

#include <opencv2/core.hpp>

#include <string>

struct ImageIntakeResult {
    bool success = false;
    cv::Mat image;
    std::string errorCode;
    std::string errorMessage;
};

class ImageIntakeService {
public:
    ImageIntakeResult decodeAndPrepare(const std::string& uploadBuffer) const;

private:
    static constexpr std::size_t kMaxUploadBytes = 10U * 1024U * 1024U;
    static constexpr int kMaxImageDimension = 1200;

    cv::Mat resizeIfNeeded(const cv::Mat& image) const;
};
