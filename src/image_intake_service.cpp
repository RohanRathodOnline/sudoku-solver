#include "image_intake_service.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

ImageIntakeResult ImageIntakeService::decodeAndPrepare(const std::string& uploadBuffer) const {
    if (uploadBuffer.empty()) {
        return {false, {}, "E_UPLOAD_EMPTY", "Uploaded image buffer is empty"};
    }

    if (uploadBuffer.size() > kMaxUploadBytes) {
        return {false, {}, "E_IMAGE_TOO_LARGE", "Uploaded image exceeds 10MB limit"};
    }

    const std::vector<unsigned char> encoded(uploadBuffer.begin(), uploadBuffer.end());
    cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        return {false, {}, "E_DECODE_FAILED", "Failed to decode uploaded image"};
    }

    decoded = resizeIfNeeded(decoded);
    return {true, decoded, {}, {}};
}

cv::Mat ImageIntakeService::resizeIfNeeded(const cv::Mat& image) const {
    if (image.empty()) {
        return {};
    }

    const int maxDimension = std::max(image.cols, image.rows);
    if (maxDimension <= kMaxImageDimension) {
        return image;
    }

    const double scale = static_cast<double>(kMaxImageDimension) / static_cast<double>(maxDimension);
    const int targetWidth = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int targetHeight = std::max(1, static_cast<int>(std::round(image.rows * scale)));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_AREA);
    return resized;
}
