#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

// Forward declarations — keeps compile times fast.
class BoardDetector;
class GridCleaner;
class CellExtractor;
class DigitSegmenter;
class DigitRecognizer;
class ImageProcessor;
class SudokuSolver;

struct PipelineStageTimings {
    double boardDetectionMs    = 0.0;
    double perspectiveWarpMs   = 0.0;
    double gridLineRemovalMs   = 0.0;
    double cellExtractionMs    = 0.0;
    double digitSegmentationMs = 0.0;
    double digitRecognitionMs  = 0.0;
    double sudokuSolvingMs     = 0.0;
    double totalMs             = 0.0;
};

struct PipelineResult {
    std::vector<std::vector<int>> detectedGrid;
    std::vector<std::vector<int>> solvedGrid;
    PipelineStageTimings          timings;
    bool                          success      = false;
    std::string                   errorMessage;
};

class VisionPipeline {
public:
    // FIX: Constructor loads the CNN model ONCE at startup.
    // Previously DigitRecognizer was a local variable inside recognizeDigits(),
    // which re-read mnist.onnx from disk on every single HTTP request — very
    // slow and the main cause of flaky digit recognition results.
    explicit VisionPipeline(const std::string& modelPath = "assets/models/mnist.onnx");
    ~VisionPipeline();

    PipelineResult process(const cv::Mat& image);

private:
    cv::Mat detectBoardAndWarp(const cv::Mat& image);
    cv::Mat removeGridLines(const cv::Mat& warpedGrid) const;
    std::vector<std::vector<cv::Mat>> extractCells(const cv::Mat& gridImage);
    std::vector<std::vector<cv::Mat>> segmentDigits(const std::vector<std::vector<cv::Mat>>& cells);
    std::vector<std::vector<int>>     recognizeDigits(const std::vector<std::vector<cv::Mat>>& cells,
                                                      const std::vector<std::vector<cv::Mat>>& segmentedDigits);
    bool solveGrid(const std::vector<std::vector<int>>& detectedGrid,
                   std::vector<std::vector<int>>&       solvedGrid) const;

    // FIX: These are now persistent members — created once, reused every call.
    std::unique_ptr<DigitRecognizer> recognizer_;
    std::unique_ptr<ImageProcessor>  fallbackRecognizer_;
};
