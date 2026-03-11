#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct DigitRecognitionMetrics {
    int predictedDigit = 0;
    std::vector<float> neighborDistances;
    int agreeingNeighbors = 0;
    bool lowConfidence = true;
};

class ImageProcessor {
public:
    cv::Mat extractSudokuGrid(const std::string& imagePath);
    cv::Mat extractSudokuGrid(const cv::Mat& inputImage);
    std::vector<std::vector<cv::Mat>> splitIntoCells(const cv::Mat& warpedGrid);
    bool detectDigit(const cv::Mat& cell, cv::Mat& normalizedDigit) const;
    int detectDigit(const cv::Mat& cell) const;
    DigitRecognitionMetrics recognizeDigitWithMetrics(const cv::Mat& cell);
    int recognizeDigit(const cv::Mat& cell);
    std::vector<std::vector<int>> extractDigitMatrix(const std::vector<std::vector<cv::Mat>>& cells);
    const std::vector<std::vector<DigitRecognitionMetrics>>& getLastRecognitionMetrics() const;
    const cv::Mat& getOriginalImage() const;
    const cv::Mat& getThresholdImage() const;

private:
    bool findLargestQuadrilateral(const std::vector<std::vector<cv::Point>>& contours,
                                  std::vector<cv::Point>& quadrilateral) const;
    std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point>& corners) const;
    void loadTemplates();

    cv::Mat originalImage_;
    cv::Mat thresholdImage_;
    std::vector<std::vector<DigitRecognitionMetrics>> lastRecognitionMetrics_;
    std::vector<cv::Mat> digitTemplates_;
    bool templatesLoaded_ = false;
};
