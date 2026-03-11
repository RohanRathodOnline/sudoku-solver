#pragma once

#include <opencv2/core.hpp>

#include <string>

struct ImageQualityReport {
    double blurScore = 0.0;
    double contrastScore = 0.0;
    double edgeDensity = 0.0;
    bool isBlurred = false;
    bool lowContrast = false;
    bool gridLikely = false;
    std::string overallQuality = "poor";
};

class ImageQualityAnalyzer {
public:
    // Computes blur score using Laplacian variance.
    // Higher score means sharper image; lower means blurrier image.
    double computeBlurScore(const cv::Mat& image) const;

    // Evaluates whether an image should be considered blurred.
    // Threshold policy:
    // - Base threshold is calibrated for a 1.0 MP image.
    // - Threshold is scaled slightly by sqrt(resolutionMP) so higher-resolution
    //   images require a bit more edge energy to be considered sharp.
    // - Scale is clamped to avoid overreacting to very small/very large images.
    bool isBlurred(double blurScore, int imageWidth, int imageHeight) const;

    // Computes a normalized contrast score from grayscale intensity distribution.
    // Score is normalized to [0, 1], where higher means higher contrast.
    double computeContrastScore(const cv::Mat& grayImage) const;

    // Returns true if normalized contrast score is below low-contrast threshold.
    bool isLowContrast(double contrastScore) const;

    // Contrast enhancement utility using CLAHE.
    // This does not modify the input image in-place; it returns an enhanced copy.
    cv::Mat enhanceContrastCLAHE(const cv::Mat& grayImage) const;

    // Computes edge density as ratio of Canny edge pixels to total pixels.
    double computeEdgeDensity(const cv::Mat& grayImage) const;

    // Heuristic check for whether edge density is likely to contain a grid.
    bool isGridLikely(double edgeDensity) const;

    // Computes image quality metrics and classifies overall quality.
    ImageQualityReport analyze(const cv::Mat& image) const;

    // Configures CLAHE clip limit used by enhanceContrastCLAHE.
    void setClaheClipLimit(double clipLimit);

    // Configures Canny thresholds used by computeEdgeDensity.
    void setCannyThresholds(double lowerThreshold, double upperThreshold);

    // Configures quality thresholds used by analyze and helper checks.
    void setQualityThresholds(double lowContrastThreshold,
                              double gridEdgeDensityLowerBound,
                              double gridEdgeDensityUpperBound,
                              double heavyBlurFactor);

private:
    double computeScaledThreshold(int imageWidth, int imageHeight) const;

    double claheClipLimit_ = 2.0;
    double cannyLowerThreshold_ = 50.0;
    double cannyUpperThreshold_ = 150.0;
    double lowContrastThreshold_ = 0.18;
    double gridEdgeDensityLowerBound_ = 0.03;
    double gridEdgeDensityUpperBound_ = 0.30;
    double heavyBlurFactor_ = 0.60;
};
