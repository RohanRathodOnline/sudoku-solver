#include "image_quality_analyzer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kBaseThresholdAt1MP = 120.0;
constexpr double kReferenceMegapixels = 1.0;
constexpr double kScaleStrength = 0.15;
constexpr double kMinScale = 0.85;
constexpr double kMaxScale = 1.25;

cv::Mat toGray8U(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    if (image.channels() == 1) {
        if (image.type() == CV_8UC1) {
            return image;
        }

        cv::Mat converted;
        image.convertTo(converted, CV_8UC1);
        return converted;
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return {};
    }

    if (gray.type() != CV_8UC1) {
        cv::Mat converted;
        gray.convertTo(converted, CV_8UC1);
        return converted;
    }

    return gray;
}
}

double ImageQualityAnalyzer::computeBlurScore(const cv::Mat& image) const {
    const cv::Mat gray = toGray8U(image);
    if (gray.empty()) {
        return 0.0;
    }

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    const double variance = stddev[0] * stddev[0];
    return std::isfinite(variance) ? variance : 0.0;
}

double ImageQualityAnalyzer::computeContrastScore(const cv::Mat& grayImage) const {
    const cv::Mat gray = toGray8U(grayImage);
    if (gray.empty()) {
        return 0.0;
    }

    const int histSize = 256;
    const float range[] = {0.0F, 256.0F};
    const float* ranges[] = {range};
    const int channels[] = {0};

    cv::Mat histogram;
    cv::calcHist(&gray, 1, channels, cv::Mat(), histogram, 1, &histSize, ranges, true, false);
    if (histogram.empty()) {
        return 0.0;
    }

    const double totalPixels = static_cast<double>(gray.total());
    if (totalPixels <= 0.0) {
        return 0.0;
    }

    double mean = 0.0;
    for (int i = 0; i < histSize; ++i) {
        const double frequency = histogram.at<float>(i);
        mean += static_cast<double>(i) * frequency;
    }
    mean /= totalPixels;

    double variance = 0.0;
    for (int i = 0; i < histSize; ++i) {
        const double frequency = histogram.at<float>(i);
        const double diff = static_cast<double>(i) - mean;
        variance += diff * diff * frequency;
    }
    variance /= totalPixels;

    const double stddev = std::sqrt(std::max(0.0, variance));
    const double normalized = std::clamp(stddev / 127.5, 0.0, 1.0);
    return std::isfinite(normalized) ? normalized : 0.0;
}

bool ImageQualityAnalyzer::isLowContrast(double contrastScore) const {
    return contrastScore < lowContrastThreshold_;
}

cv::Mat ImageQualityAnalyzer::enhanceContrastCLAHE(const cv::Mat& grayImage) const {
    const cv::Mat gray = toGray8U(grayImage);
    if (gray.empty()) {
        return {};
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(std::max(0.1, claheClipLimit_));
    clahe->setTilesGridSize(cv::Size(8, 8));

    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    return enhanced;
}

double ImageQualityAnalyzer::computeEdgeDensity(const cv::Mat& grayImage) const {
    const cv::Mat gray = toGray8U(grayImage);
    if (gray.empty()) {
        return 0.0;
    }

    const double lower = std::max(0.0, cannyLowerThreshold_);
    const double upper = std::max(lower + 1.0, cannyUpperThreshold_);

    cv::Mat edges;
    cv::Canny(gray, edges, lower, upper);

    const double totalPixels = static_cast<double>(edges.total());
    if (totalPixels <= 0.0) {
        return 0.0;
    }

    const double edgePixels = static_cast<double>(cv::countNonZero(edges));
    return edgePixels / totalPixels;
}

bool ImageQualityAnalyzer::isGridLikely(double edgeDensity) const {
    return edgeDensity >= gridEdgeDensityLowerBound_ && edgeDensity <= gridEdgeDensityUpperBound_;
}

ImageQualityReport ImageQualityAnalyzer::analyze(const cv::Mat& image) const {
    ImageQualityReport report;

    report.blurScore = computeBlurScore(image);
    report.contrastScore = computeContrastScore(image);
    report.edgeDensity = computeEdgeDensity(image);

    const bool invalidDimensions = image.empty() || image.cols <= 0 || image.rows <= 0;
    report.isBlurred = invalidDimensions || isBlurred(report.blurScore, image.cols, image.rows);
    report.lowContrast = isLowContrast(report.contrastScore);
    report.gridLikely = isGridLikely(report.edgeDensity);

    const double blurThreshold =
        invalidDimensions ? std::numeric_limits<double>::infinity() : computeScaledThreshold(image.cols, image.rows);
    const bool heavilyBlurred = report.blurScore < (blurThreshold * std::max(0.1, heavyBlurFactor_));

    if (heavilyBlurred || !report.gridLikely) {
        report.overallQuality = "poor";
        return report;
    }

    const int minorIssueCount = (report.isBlurred ? 1 : 0) + (report.lowContrast ? 1 : 0);
    report.overallQuality = (minorIssueCount > 0) ? "acceptable" : "good";
    return report;
}

void ImageQualityAnalyzer::setClaheClipLimit(double clipLimit) {
    claheClipLimit_ = std::max(0.1, clipLimit);
}

void ImageQualityAnalyzer::setCannyThresholds(double lowerThreshold, double upperThreshold) {
    cannyLowerThreshold_ = std::max(0.0, lowerThreshold);
    cannyUpperThreshold_ = std::max(cannyLowerThreshold_ + 1.0, upperThreshold);
}

void ImageQualityAnalyzer::setQualityThresholds(double lowContrastThreshold,
                                                double gridEdgeDensityLowerBound,
                                                double gridEdgeDensityUpperBound,
                                                double heavyBlurFactor) {
    lowContrastThreshold_ = std::clamp(lowContrastThreshold, 0.0, 1.0);
    gridEdgeDensityLowerBound_ = std::clamp(gridEdgeDensityLowerBound, 0.0, 1.0);
    gridEdgeDensityUpperBound_ = std::clamp(gridEdgeDensityUpperBound, gridEdgeDensityLowerBound_, 1.0);
    heavyBlurFactor_ = std::clamp(heavyBlurFactor, 0.1, 1.0);
}

bool ImageQualityAnalyzer::isBlurred(double blurScore, int imageWidth, int imageHeight) const {
    if (imageWidth <= 0 || imageHeight <= 0) {
        return true;
    }

    const double threshold = computeScaledThreshold(imageWidth, imageHeight);
    return blurScore < threshold;
}

double ImageQualityAnalyzer::computeScaledThreshold(int imageWidth, int imageHeight) const {
    const double megapixels = static_cast<double>(imageWidth) * static_cast<double>(imageHeight) / 1'000'000.0;
    const double normalizedScale = std::sqrt(std::max(0.01, megapixels / kReferenceMegapixels));

    const double scale = std::clamp(
        1.0 + kScaleStrength * (normalizedScale - 1.0),
        kMinScale,
        kMaxScale
    );

    return kBaseThresholdAt1MP * scale;
}
