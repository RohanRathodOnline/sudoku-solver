#include "digit_segmenter.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace {
constexpr int kNormalizedDigitSize = 28;
constexpr int kInnerDigitSize = 20;
}

bool DigitSegmenter::extractDigit(const cv::Mat& cell, cv::Mat& digit) const {
    digit.release();
    if (cell.empty()) {
        return false;
    }

    cv::Mat gray;
    if (cell.channels() == 1) {
        gray = cell.clone();
    } else {
        cv::cvtColor(cell, gray, cv::COLOR_BGR2GRAY);
    }

    const int minSide = std::min(gray.rows, gray.cols);
    const int border = std::max(1, minSide / 10);
    if (gray.cols <= border * 2 || gray.rows <= border * 2) {
        return false;
    }

    cv::Mat inner = gray(cv::Rect(border, border, gray.cols - border * 2, gray.rows - border * 2)).clone();

    cv::Mat binary;
    cv::adaptiveThreshold(
        inner,
        binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        11,
        2
    );

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int labelCount = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    if (labelCount <= 1) {
        return false;
    }

    const int cellArea = binary.rows * binary.cols;
    const int minComponentArea = std::max(6, static_cast<int>(std::round(cellArea * 0.004)));

    int largestLabel = -1;
    int largestArea = 0;
    for (int label = 1; label < labelCount; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minComponentArea) {
            continue;
        }

        if (area > largestArea) {
            largestArea = area;
            largestLabel = label;
        }
    }

    if (largestLabel < 0) {
        return false;
    }

    const int x = stats.at<int>(largestLabel, cv::CC_STAT_LEFT);
    const int y = stats.at<int>(largestLabel, cv::CC_STAT_TOP);
    const int width = stats.at<int>(largestLabel, cv::CC_STAT_WIDTH);
    const int height = stats.at<int>(largestLabel, cv::CC_STAT_HEIGHT);
    if (width <= 0 || height <= 0) {
        return false;
    }

    cv::Mat componentMask = (labels == largestLabel);
    componentMask.convertTo(componentMask, CV_8UC1, 255.0);

    const cv::Rect bbox(x, y, width, height);
    digit = normalizeTo28x28(componentMask(bbox));
    return !digit.empty();
}

cv::Mat DigitSegmenter::normalizeTo28x28(const cv::Mat& componentMask) const {
    if (componentMask.empty()) {
        return {};
    }

    std::vector<cv::Point> points;
    cv::findNonZero(componentMask, points);
    if (points.empty()) {
        return {};
    }

    const cv::Rect bbox = cv::boundingRect(points);
    cv::Mat cropped = componentMask(bbox);

    const float scale = std::min(
        static_cast<float>(kInnerDigitSize) / static_cast<float>(std::max(1, bbox.width)),
        static_cast<float>(kInnerDigitSize) / static_cast<float>(std::max(1, bbox.height))
    );

    const int targetWidth = std::max(1, static_cast<int>(std::round(static_cast<float>(bbox.width) * scale)));
    const int targetHeight = std::max(1, static_cast<int>(std::round(static_cast<float>(bbox.height) * scale)));

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_AREA);

    cv::Mat canvas(kNormalizedDigitSize, kNormalizedDigitSize, CV_8UC1, cv::Scalar(0));
    const int offsetX = (kNormalizedDigitSize - targetWidth) / 2;
    const int offsetY = (kNormalizedDigitSize - targetHeight) / 2;
    resized.copyTo(canvas(cv::Rect(offsetX, offsetY, targetWidth, targetHeight)));

    return canvas;
}
