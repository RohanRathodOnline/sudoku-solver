#include "board_detector.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

namespace {
constexpr int kWarpedSudokuSize = 450;
} // namespace

bool BoardDetector::detectBoard(const cv::Mat& image, cv::Mat& warpedBoard) {
    warpedBoard.release();

    if (image.empty()) {
        return false;
    }

    try {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Enhance local contrast so faint printed grids still get detected.
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat contrastEnhanced;
        clahe->apply(gray, contrastEnhanced);

        cv::Mat blurred;
        cv::GaussianBlur(contrastEnhanced, blurred, cv::Size(7, 7), 0.0);

        cv::Mat thresholded;
        cv::adaptiveThreshold(
            blurred,
            thresholded,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY,
            11,
            2
        );

        cv::bitwise_not(thresholded, thresholded);

        const cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE, closeKernel);

        cv::Mat edges;
        cv::Canny(thresholded, edges, 50, 150);

        std::vector<cv::Point> quadrilateral;

#ifdef _DEBUG
        cv::Mat debugContour;
        if (!detectSudokuBoardCorners(edges, quadrilateral, &debugContour)) {
            // FIX: Return false — do NOT silently fall back to a random crop.
            // The caller (VisionPipeline) will surface a clear error to the
            // frontend instead of continuing with garbage pixel data.
            return false;
        }
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(fs::path("debug_output"), ec);
        cv::imwrite((fs::path("debug_output") / "detected_sudoku_contour.png").string(), debugContour);
#else
        if (!detectSudokuBoardCorners(edges, quadrilateral, nullptr)) {
            return false;
        }
#endif

        // Reject detection if the quad covers less than 35% of the image —
        // that means we found something other than the main Sudoku board.
        // FIX: Return false instead of silently cropping the image center.
        const double quadArea  = std::abs(cv::contourArea(quadrilateral));
        const double imageArea = static_cast<double>(gray.rows * gray.cols);
        if (quadArea < imageArea * 0.35) {
            return false;
        }

        const std::vector<cv::Point2f> ordered = orderCorners(quadrilateral);
        if (ordered.size() != 4) {
            return false;
        }

        const std::vector<cv::Point2f> destination = {
            {0.0F,                               0.0F},
            {static_cast<float>(kWarpedSudokuSize - 1), 0.0F},
            {static_cast<float>(kWarpedSudokuSize - 1), static_cast<float>(kWarpedSudokuSize - 1)},
            {0.0F,                               static_cast<float>(kWarpedSudokuSize - 1)}
        };

        cv::Mat transform = cv::getPerspectiveTransform(ordered, destination);
        cv::warpPerspective(gray, warpedBoard, transform, cv::Size(kWarpedSudokuSize, kWarpedSudokuSize));

        return !warpedBoard.empty();

    } catch (const cv::Exception&) {
        return false;
    }
}

bool BoardDetector::detectSudokuBoardCorners(const cv::Mat& preprocessedImage,
                                             std::vector<cv::Point>& boardCorners,
                                             cv::Mat* debugVisualization) const {
    boardCorners.clear();
    if (preprocessedImage.empty()) {
        return false;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(preprocessedImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0.0;
    std::vector<cv::Point> bestApprox;

    for (const auto& contour : contours) {
        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) {
            continue;
        }

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        if (approx.size() != 4 || !cv::isContourConvex(approx)) {
            continue;
        }

        const double area = std::abs(cv::contourArea(approx));
        if (area > maxArea) {
            maxArea = area;
            bestApprox = approx;
        }
    }

    if (bestApprox.size() != 4) {
        return false;
    }

    boardCorners = bestApprox;

    if (debugVisualization != nullptr) {
        if (preprocessedImage.channels() == 1) {
            cv::cvtColor(preprocessedImage, *debugVisualization, cv::COLOR_GRAY2BGR);
        } else {
            *debugVisualization = preprocessedImage.clone();
        }
        cv::drawContours(*debugVisualization,
                         std::vector<std::vector<cv::Point>>{boardCorners},
                         -1, cv::Scalar(0, 255, 0), 2);
        for (const auto& p : boardCorners) {
            cv::circle(*debugVisualization, p, 5, cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }

    return true;
}

bool BoardDetector::findLargestQuadrilateral(const std::vector<std::vector<cv::Point>>& contours,
                                             std::vector<cv::Point>& quadrilateral) const {
    double largestArea = 0.0;

    for (const auto& contour : contours) {
        double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) {
            continue;
        }

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        if (approx.size() != 4 || !cv::isContourConvex(approx)) {
            continue;
        }

        double area = std::abs(cv::contourArea(approx));
        if (area > largestArea) {
            largestArea = area;
            quadrilateral = approx;
        }
    }

    return !quadrilateral.empty();
}

std::vector<cv::Point2f> BoardDetector::orderCorners(const std::vector<cv::Point>& corners) const {
    if (corners.size() != 4) {
        return {};
    }

    std::vector<cv::Point2f> ordered(4);

    float minSum  = std::numeric_limits<float>::max();
    float maxSum  = std::numeric_limits<float>::lowest();
    float minDiff = std::numeric_limits<float>::max();
    float maxDiff = std::numeric_limits<float>::lowest();

    for (const auto& point : corners) {
        const float x    = static_cast<float>(point.x);
        const float y    = static_cast<float>(point.y);
        const float sum  = x + y;
        const float diff = y - x;

        if (sum < minSum)  { minSum  = sum;  ordered[0] = {x, y}; } // top-left
        if (sum > maxSum)  { maxSum  = sum;  ordered[2] = {x, y}; } // bottom-right
        if (diff < minDiff){ minDiff = diff; ordered[1] = {x, y}; } // top-right
        if (diff > maxDiff){ maxDiff = diff; ordered[3] = {x, y}; } // bottom-left
    }

    return ordered;
}
