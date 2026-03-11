#include "image_processor.h"
#include "board_detector.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>        // Machine learning module
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {             
constexpr int kNormalizedDigitSize = 28;
constexpr int kInnerDigitSize = 20;

constexpr double kMinContourAreaRatio = 0.008;
constexpr double kMinConnectedComponentAreaRatio = 0.02;
constexpr double kMinDigitBoundingBoxAreaRatio = 0.015;
constexpr float kTemplateConfidenceMin = 0.60F;
constexpr float kAmbiguityGapMin = 0.02F;
constexpr int kMinAgreeingNeighborsForHighConfidence = 2;
constexpr float kLowConfidenceAverageDistanceThreshold = 180.0F;
constexpr float kLowConfidenceMaxDistanceThreshold = 260.0F;
constexpr double kLowMeanIntensityThreshold = 80.0;
constexpr double kClaheClipLimit = 2.0;
constexpr int kClaheTileGridSize = 8;
constexpr double kAdaptiveBlockFractionOfCellWidth = 0.5;
constexpr int kAdaptiveBlockSizeMinimum = 3;
constexpr double kAdaptiveThresholdOffset = 2.0;
constexpr int kSmallCellThreshold = 25;
constexpr int kExtremeSmallCellThreshold = 15;
constexpr int kTargetNormalizedCellSize = 40;
constexpr int kMinNormalizedDigitInk = 16;
constexpr int kWarpedSudokuSize = 450;
constexpr float kCnnPredictionConfidenceThreshold = 0.65F;
constexpr const char* kDigitCnnModelPath = "assets/models/mnist.onnx";

std::filesystem::path getExecutableDirectory() {
#ifdef _WIN32
    char buffer[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (length > 0 && length < MAX_PATH) {
        return std::filesystem::path(buffer).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

std::filesystem::path resolveModelPath(const std::filesystem::path& relativeModelPath) {
    const std::filesystem::path cwdCandidate = std::filesystem::absolute(relativeModelPath);
    if (std::filesystem::exists(cwdCandidate)) {
        return cwdCandidate;
    }

    const std::filesystem::path exeCandidate = getExecutableDirectory() / relativeModelPath;
    if (std::filesystem::exists(exeCandidate)) {
        return exeCandidate;
    }

    return cwdCandidate;
}

struct TemplateMatchResult {
    int digit = 0;
    float bestScore = -1.0F;
    float secondBestScore = -1.0F;
};

// Parameter selection strategy for per-cell thresholding:
// 1) Measure mean intensity for each cell.
// 2) If cell is dark (mean < kLowMeanIntensityThreshold), apply CLAHE to recover local contrast.
// 3) Choose adaptiveThreshold block size from cell width (~50%), rounded to nearest odd,
//    and clamped to a valid odd range based on current cell dimensions.
int computeCellAdaptiveBlockSize(int cellWidth, int cellHeight) {
    if (cellWidth <= 0 || cellHeight <= 0) {
        return kAdaptiveBlockSizeMinimum;
    }

    int blockSize = static_cast<int>(std::round(static_cast<double>(cellWidth) * kAdaptiveBlockFractionOfCellWidth));
    if (blockSize % 2 == 0) {
        ++blockSize;
    }

    int maxValidOdd = std::min(cellWidth, cellHeight);
    if (maxValidOdd % 2 == 0) {
        --maxValidOdd;
    }

    if (maxValidOdd < kAdaptiveBlockSizeMinimum) {
        return kAdaptiveBlockSizeMinimum;
    }

    blockSize = std::clamp(blockSize, kAdaptiveBlockSizeMinimum, maxValidOdd);
    if (blockSize % 2 == 0) {
        blockSize = std::max(kAdaptiveBlockSizeMinimum, blockSize - 1);
    }

    return blockSize;
}

cv::Mat prepareCellForAdaptiveThreshold(const cv::Mat& grayCell, double& meanIntensity) {
    meanIntensity = cv::mean(grayCell)[0];

    if (meanIntensity >= kLowMeanIntensityThreshold) {
        return grayCell;
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(kClaheClipLimit, cv::Size(kClaheTileGridSize, kClaheTileGridSize));
    cv::Mat enhanced;
    clahe->apply(grayCell, enhanced);
    return enhanced;
}

bool isSudokuCellEmpty(const cv::Mat& cell) {
    if (cell.empty()) {
        return true;
    }

    cv::Mat gray;
    if (cell.channels() == 1) {
        gray = cell;
    } else {
        cv::cvtColor(cell, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::adaptiveThreshold(
        gray,
        binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        11,
        2
    );

    const int whitePixels = cv::countNonZero(binary);
    const int pixelThreshold = std::max(6, static_cast<int>(std::round(binary.rows * binary.cols * 0.004)));
    return whitePixels < pixelThreshold;
}

cv::Mat normalizeCellResolution(const cv::Mat& cell, int row, int col) {
    if (cell.empty() || cell.cols <= 0 || cell.rows <= 0) {
        return {};
    }

    const int cellWidth = cell.cols;
    const int cellHeight = cell.rows;

    if (cellWidth < kExtremeSmallCellThreshold || cellHeight < kExtremeSmallCellThreshold) {
#ifdef _DEBUG
        std::cout << "[DEBUG] Cell too small to process at (" << row << ", " << col
                  << ") size=" << cellWidth << "x" << cellHeight << std::endl;
#endif
        return {};
    }

    if (cellWidth >= kSmallCellThreshold && cellHeight >= kSmallCellThreshold) {
        return cell;
    }

#ifdef _DEBUG
    std::cout << "[DEBUG] Cell resolution too small at (" << row << ", " << col << ")" << std::endl;
#endif

    const double scale = std::max(
        static_cast<double>(kTargetNormalizedCellSize) / static_cast<double>(cellWidth),
        static_cast<double>(kTargetNormalizedCellSize) / static_cast<double>(cellHeight)
    );

    const int targetWidth = std::max(1, static_cast<int>(std::round(static_cast<double>(cellWidth) * scale)));
    const int targetHeight = std::max(1, static_cast<int>(std::round(static_cast<double>(cellHeight) * scale)));

    cv::Mat upscaled;
    cv::resize(cell, upscaled, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_CUBIC);
    return upscaled;
}

#ifdef _DEBUG
thread_local int gCurrentDebugRow = -1;
thread_local int gCurrentDebugCol = -1;

void setCurrentDebugCell(int row, int col) {
    gCurrentDebugRow = row;
    gCurrentDebugCol = col;
}

void saveCellDebugImages(const cv::Mat& cellOriginal,
                         const cv::Mat& cellThreshold,
                         const cv::Mat& cellMorph,
                         const cv::Mat& cellDigitMask) {
    if (gCurrentDebugRow < 0 || gCurrentDebugCol < 0) {
        return;
    }

    namespace fs = std::filesystem;
    const fs::path outputDir = fs::path("debug_output") /
                               (std::to_string(gCurrentDebugRow) + "_" + std::to_string(gCurrentDebugCol));

    std::error_code ec;
    fs::create_directories(outputDir, ec);

    if (!cellOriginal.empty()) {
        cv::imwrite((outputDir / "cell_original.png").string(), cellOriginal);
    }
    if (!cellThreshold.empty()) {
        cv::imwrite((outputDir / "cell_threshold.png").string(), cellThreshold);
    }
    if (!cellMorph.empty()) {
        cv::imwrite((outputDir / "cell_morph.png").string(), cellMorph);
    }
    if (!cellDigitMask.empty()) {
        cv::imwrite((outputDir / "cell_digit_mask.png").string(), cellDigitMask);
    }
}
#endif

bool detectSudokuBoardCorners(const cv::Mat& preprocessedImage,
                              std::vector<cv::Point>& boardCorners,
                              cv::Mat* debugVisualization = nullptr) {
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

        cv::drawContours(*debugVisualization, std::vector<std::vector<cv::Point>>{boardCorners}, -1, cv::Scalar(0, 255, 0), 2);
        for (const auto& p : boardCorners) {
            cv::circle(*debugVisualization, p, 5, cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }

    return true;
}

std::vector<cv::Mat> splitSudokuBoardInto81Cells(const cv::Mat& warpedSudokuImage) {
    if (warpedSudokuImage.empty()) {
        return {};
    }

    cv::Mat board;
    if (warpedSudokuImage.rows != kWarpedSudokuSize || warpedSudokuImage.cols != kWarpedSudokuSize) {
        cv::resize(
            warpedSudokuImage,
            board,
            cv::Size(kWarpedSudokuSize, kWarpedSudokuSize),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    } else {
        board = warpedSudokuImage;
    }

    const int cellSize = kWarpedSudokuSize / 9;
    std::vector<cv::Mat> cells;
    cells.reserve(81);

    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            const int x = col * cellSize;
            const int y = row * cellSize;
            const int width = (col == 8) ? (kWarpedSudokuSize - x) : cellSize;
            const int height = (row == 8) ? (kWarpedSudokuSize - y) : cellSize;

            cells.push_back(board(cv::Rect(x, y, width, height)).clone());
        }
    }

#ifdef _DEBUG
    for (size_t i = 0; i < std::min<size_t>(6, cells.size()); ++i) {
        cv::imshow("debug_cell_" + std::to_string(i), cells[i]);
    }
    cv::waitKey(1);
#endif

    return cells;
}

bool convertRecognizedDigitsToMatrix(const std::vector<int>& digits, int sudoku[9][9]) {
    if (digits.size() != 81) {
        return false;
    }

    int index = 0;
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            sudoku[row][col] = digits[index++];
        }
    }

    std::cout << "\n===== SUDOKU 9x9 MATRIX =====\n";
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            std::cout << sudoku[row][col] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "=============================\n";

    return true;
}

cv::Mat normalizeDigitToCanvas(const cv::Mat& digitMask) {
    if (digitMask.empty()) {
        return {};
    }

    cv::Mat gray;
    if (digitMask.channels() == 1) {
        gray = digitMask.clone();
    } else {
        cv::cvtColor(digitMask, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    if (gray.type() == CV_8UC1) {
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } else {
        gray.convertTo(binary, CV_8UC1);
        cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    std::vector<cv::Point> points;
    cv::findNonZero(binary, points);
    if (points.empty()) {
        return {};
    }

    const cv::Rect bbox = cv::boundingRect(points);
    if (bbox.width <= 0 || bbox.height <= 0) {
        return {};
    }

    cv::Mat cropped = binary(bbox);

    const float scale = std::min(
        static_cast<float>(kInnerDigitSize) / static_cast<float>(bbox.width),
        static_cast<float>(kInnerDigitSize) / static_cast<float>(bbox.height)
    );

    const int targetWidth = std::max(1, static_cast<int>(std::round(bbox.width * scale)));
    const int targetHeight = std::max(1, static_cast<int>(std::round(bbox.height * scale)));

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_AREA);

    cv::Mat canvas(kNormalizedDigitSize, kNormalizedDigitSize, CV_8UC1, cv::Scalar(0));
    const int offsetX = (kNormalizedDigitSize - targetWidth) / 2;
    const int offsetY = (kNormalizedDigitSize - targetHeight) / 2;

    resized.copyTo(canvas(cv::Rect(offsetX, offsetY, targetWidth, targetHeight)));
    return canvas;
}

cv::Mat extractDigitFromSudokuCell(const cv::Mat& cellImage) {
    if (cellImage.empty()) {
        return {};
    }

    cv::Mat gray;
    if (cellImage.channels() == 1) {
        gray = cellImage.clone();
    } else {
        cv::cvtColor(cellImage, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::adaptiveThreshold(
        gray,
        binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        11,
        2
    );

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int largestContourIndex = -1;
    double largestContourArea = 0.0;
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > largestContourArea) {
            largestContourArea = area;
            largestContourIndex = i;
        }
    }

    if (largestContourIndex < 0) {
        return {};
    }

    const cv::Rect bbox = cv::boundingRect(contours[largestContourIndex]);
    if (bbox.width <= 0 || bbox.height <= 0) {
        return {};
    }

    cv::Mat digitCropped = binary(bbox).clone();

    const int squareSide = std::max(digitCropped.rows, digitCropped.cols);
    cv::Mat squareCanvas(squareSide, squareSide, CV_8UC1, cv::Scalar(0));
    const int offsetX = (squareSide - digitCropped.cols) / 2;
    const int offsetY = (squareSide - digitCropped.rows) / 2;
    digitCropped.copyTo(squareCanvas(cv::Rect(offsetX, offsetY, digitCropped.cols, digitCropped.rows)));

    cv::Mat normalized;
    cv::resize(squareCanvas, normalized, cv::Size(kNormalizedDigitSize, kNormalizedDigitSize), 0, 0, cv::INTER_AREA);
    return normalized;
}

bool extractLargestConnectedDigit(const cv::Mat& binary, cv::Mat& normalizedDigit);

bool preprocessCellForTemplateMatching(
    const cv::Mat& inputGray,
    cv::Mat& normalizedDigit
#ifdef _DEBUG
    , cv::Mat* thresholdDebug,
    cv::Mat* morphDebug
#endif
) {
    if (inputGray.empty()) {
        return false;
    }

    cv::Mat gray;
    if (inputGray.channels() == 1) {
        gray = inputGray.clone();
    } else {
        cv::cvtColor(inputGray, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);

    double meanIntensity = 0.0;
    const cv::Mat thresholdInput = prepareCellForAdaptiveThreshold(blurred, meanIntensity);
    const int adaptiveBlockSize = computeCellAdaptiveBlockSize(thresholdInput.cols, thresholdInput.rows);

    cv::Mat binary;
    cv::adaptiveThreshold(
        thresholdInput,
        binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        adaptiveBlockSize,
        kAdaptiveThresholdOffset
    );

    const cv::Mat adaptiveBinary = binary.clone();

    cv::Mat distanceMap;
    cv::distanceTransform(binary, distanceMap, cv::DIST_L2, 3);

    cv::Mat distanceMapNormalized;
    cv::normalize(distanceMap, distanceMapNormalized, 0.0, 255.0, cv::NORM_MINMAX);

    cv::Mat distanceMask;
    cv::threshold(distanceMapNormalized, distanceMask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    distanceMask.convertTo(distanceMask, CV_8UC1);

    const int cellAreaFromMask = std::max(1, distanceMask.rows * distanceMask.cols);
    const int distanceInk = cv::countNonZero(distanceMask);
    if (distanceInk >= static_cast<int>(cellAreaFromMask * 0.003)) {
        binary = distanceMask;
    } else {
        binary = adaptiveBinary;
    }

    const int cellHeight = std::max(1, binary.rows);
    const int cellWidth = std::max(1, binary.cols);

    cv::Mat verticalLines;
    cv::Mat horizontalLines;
    const cv::Mat verticalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, std::max(1, cellHeight / 4)));
    const cv::Mat horizontalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, cellWidth / 4), 1));

    cv::erode(binary, verticalLines, verticalKernel);
    cv::dilate(verticalLines, verticalLines, verticalKernel);

    cv::erode(binary, horizontalLines, horizontalKernel);
    cv::dilate(horizontalLines, horizontalLines, horizontalKernel);

    cv::Mat gridLinesMask;
    cv::bitwise_or(verticalLines, horizontalLines, gridLinesMask);
    const cv::Mat beforeGridRemoval = binary.clone();
    cv::subtract(binary, gridLinesMask, binary);

    const int afterGridRemovalInk = cv::countNonZero(binary);
    if (afterGridRemovalInk < static_cast<int>(cellAreaFromMask * 0.0025)) {
        binary = beforeGridRemoval;
    }

    const cv::Mat binaryBeforeMorph = binary.clone();

#ifdef _DEBUG
    if (thresholdDebug != nullptr) {
        *thresholdDebug = binary.clone();
    }
#endif

    const int cellArea = binary.rows * binary.cols;
    if (cellArea <= 0) {
        return false;
    }

    cv::Mat binaryOtsu;
    cv::threshold(thresholdInput, binaryOtsu, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int adaptiveInk = cv::countNonZero(binary);
    if (adaptiveInk < static_cast<int>(cellArea * 0.01) ||
        adaptiveInk > static_cast<int>(cellArea * 0.85)) {
        binary = binaryOtsu;
    }

    if (cv::countNonZero(binary) > static_cast<int>(cellArea * 0.70)) {
        cv::bitwise_not(binary, binary);
    }

    const int finalInk = cv::countNonZero(binary);
    if (finalInk < static_cast<int>(cellArea * 0.004)) {
        return false;
    }

    const int kernelSide = std::max(2, std::min(binary.rows, binary.cols) / 16);
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSide, kernelSide));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, openKernel);

    if (cv::countNonZero(binary) < static_cast<int>(cellArea * 0.004)) {
        binary = binaryBeforeMorph;
    }

#ifdef _DEBUG
    if (morphDebug != nullptr) {
        *morphDebug = binary.clone();
    }
#endif

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double minContourArea = std::max(10.0, static_cast<double>(cellArea) * kMinContourAreaRatio);
    const int borderPadding = std::max(1, std::min(binary.rows, binary.cols) / 18);

    int bestContourIndex = -1;
    double bestContourArea = 0.0;

    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area < minContourArea) {
            continue;
        }

        const cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.width <= 1 || rect.height <= 1) {
            continue;
        }

        const bool touchesBorder =
            rect.x <= borderPadding ||
            rect.y <= borderPadding ||
            (rect.x + rect.width) >= (binary.cols - borderPadding) ||
            (rect.y + rect.height) >= (binary.rows - borderPadding);

        if (touchesBorder) {
            const double aspect = static_cast<double>(rect.width) / static_cast<double>(rect.height);
            const bool likelyGridLine = aspect > 5.0 || aspect < 0.2;
            if (likelyGridLine) {
                continue;
            }
        }

        if (area > bestContourArea) {
            bestContourArea = area;
            bestContourIndex = i;
        }
    }

    if (bestContourIndex < 0) {
        return extractLargestConnectedDigit(binary, normalizedDigit);
    }

    cv::Mat contourMask(binary.size(), CV_8UC1, cv::Scalar(0));
    cv::drawContours(contourMask, contours, bestContourIndex, cv::Scalar(255), cv::FILLED);

    cv::Mat isolated;
    cv::bitwise_and(binary, contourMask, isolated);

    const cv::Rect digitRect = cv::boundingRect(contours[bestContourIndex]);
    if (digitRect.width <= 1 || digitRect.height <= 1) {
        return false;
    }

    normalizedDigit = extractDigitFromSudokuCell(isolated(digitRect));
    return !normalizedDigit.empty() && cv::countNonZero(normalizedDigit) >= kMinNormalizedDigitInk;
}

bool extractLargestConnectedDigit(const cv::Mat& binary, cv::Mat& normalizedDigit) {
    normalizedDigit.release();
    if (binary.empty() || binary.type() != CV_8UC1) {
        return false;
    }

    const int cellArea = binary.rows * binary.cols;
    if (cellArea <= 0) {
        return false;
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int labelCount = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    if (labelCount <= 1) {
        return false;
    }

    const int minComponentArea = std::max(1, static_cast<int>(std::round(cellArea * kMinConnectedComponentAreaRatio)));

    cv::Mat filtered(binary.size(), CV_8UC1, cv::Scalar(0));
    for (int label = 1; label < labelCount; ++label) {
        const int componentArea = stats.at<int>(label, cv::CC_STAT_AREA);
        if (componentArea < minComponentArea) {
            continue;
        }

        cv::Mat componentMask = (labels == label);
        filtered.setTo(255, componentMask);
    }

    cv::Mat filteredLabels;
    cv::Mat filteredStats;
    cv::Mat filteredCentroids;
    const int filteredLabelCount =
        cv::connectedComponentsWithStats(filtered, filteredLabels, filteredStats, filteredCentroids, 8, CV_32S);
    if (filteredLabelCount <= 1) {
        return false;
    }

    int largestLabel = -1;
    int largestArea = 0;
    for (int label = 1; label < filteredLabelCount; ++label) {
        const int area = filteredStats.at<int>(label, cv::CC_STAT_AREA);
        if (area > largestArea) {
            largestArea = area;
            largestLabel = label;
        }
    }

    if (largestLabel <= 0) {
        return false;
    }

    const int x = filteredStats.at<int>(largestLabel, cv::CC_STAT_LEFT);
    const int y = filteredStats.at<int>(largestLabel, cv::CC_STAT_TOP);
    const int width = filteredStats.at<int>(largestLabel, cv::CC_STAT_WIDTH);
    const int height = filteredStats.at<int>(largestLabel, cv::CC_STAT_HEIGHT);

    if (width <= 0 || height <= 0) {
        return false;
    }

    const int bboxArea = width * height;
    if (bboxArea < static_cast<int>(std::round(cellArea * kMinDigitBoundingBoxAreaRatio))) {
        return false;
    }

    const cv::Rect bbox(x, y, width, height);
    cv::Mat largestMask = (filteredLabels == largestLabel);
    largestMask.convertTo(largestMask, CV_8UC1, 255.0);

    normalizedDigit = normalizeDigitToCanvas(largestMask(bbox));
    return !normalizedDigit.empty();
}

cv::Mat prepareTemplateForMatching(const cv::Mat& rawTemplate) {
    if (rawTemplate.empty()) {
        return {};
    }

    cv::Mat gray;
    if (rawTemplate.channels() == 1) {
        gray = rawTemplate.clone();
    } else {
        cv::cvtColor(rawTemplate, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::adaptiveThreshold(
        gray,
        binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        11,
        2
    );

    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, openKernel);

    return normalizeDigitToCanvas(binary);
}

TemplateMatchResult matchAgainstTemplates(const cv::Mat& normalizedDigit, const std::vector<cv::Mat>& templates) {
    TemplateMatchResult result;

    for (int i = 0; i < static_cast<int>(templates.size()); ++i) {
        if (templates[i].empty()) {
            continue;
        }

        cv::Mat templ = templates[i];
        if (templ.size() != normalizedDigit.size()) {
            cv::resize(templ, templ, normalizedDigit.size(), 0, 0, cv::INTER_AREA);
        }

        cv::Mat response;
        cv::matchTemplate(normalizedDigit, templ, response, cv::TM_CCOEFF_NORMED);
        const float score = response.at<float>(0, 0);

        if (score > result.bestScore) {
            result.secondBestScore = result.bestScore;
            result.bestScore = score;
            result.digit = i + 1;
        } else if (score > result.secondBestScore) {
            result.secondBestScore = score;
        }
    }

    return result;
}

bool extractDigitMask(const cv::Mat& grayCell, cv::Mat& normalizedDigit) {
#ifdef _DEBUG
    cv::Mat thresholdDebug;
    cv::Mat morphDebug;
    const bool success = preprocessCellForTemplateMatching(grayCell, normalizedDigit, &thresholdDebug, &morphDebug);
    saveCellDebugImages(grayCell, thresholdDebug, morphDebug, normalizedDigit);
    return success;
#else
    return preprocessCellForTemplateMatching(grayCell, normalizedDigit);
#endif
}

cv::Mat flattenSample(const cv::Mat& normalizedDigit) {
    cv::Mat floatSample;
    normalizedDigit.reshape(1, 1).convertTo(floatSample, CV_32F, 1.0 / 255.0);
    return floatSample;
}

bool predictDigitWithCnn(const cv::Mat& normalizedDigit, int& predictedDigit, float& confidence) {
    predictedDigit = 0;
    confidence = 0.0F;

    static cv::dnn::Net net;
    static bool loadAttempted = false;
    if (!loadAttempted) {
        loadAttempted = true;
        const std::filesystem::path resolvedModelPath = resolveModelPath(kDigitCnnModelPath);
        std::cout << "Loading CNN model from: " << resolvedModelPath.string() << std::endl;

        if (!std::filesystem::exists(resolvedModelPath)) {
            std::cerr << "ERROR: CNN model not found at " << resolvedModelPath.string() << std::endl;
            std::cerr << "Ensure assets/models/mnist.onnx exists in the runtime directory." << std::endl;
            return false;
        }

        try {
            net = cv::dnn::readNet(resolvedModelPath.string());
            if (!net.empty()) {
                std::cout << "CNN model loaded successfully." << std::endl;
            } else {
                std::cerr << "ERROR: Failed to load CNN model from " << resolvedModelPath.string() << std::endl;
                return false;
            }
        } catch (const cv::Exception&) {
            std::cerr << "ERROR: OpenCV failed to read CNN model at " << resolvedModelPath.string() << std::endl;
            return false;
        }
    }

    if (net.empty()) {
        return false;
    }

    cv::Mat inputGray;
    if (normalizedDigit.channels() == 1) {
        inputGray = normalizedDigit;
    } else {
        cv::cvtColor(normalizedDigit, inputGray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        inputGray,
        1.0 / 255.0,
        cv::Size(kNormalizedDigitSize, kNormalizedDigitSize),
        cv::Scalar(),
        false,
        false,
        CV_32F
    );

    net.setInput(blob);
    cv::Mat output = net.forward();
    if (output.empty()) {
        return false;
    }

    cv::Mat scores = output.reshape(1, 1);
    cv::Point classIdPoint;
    double maxScore = 0.0;
    cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);

    predictedDigit = classIdPoint.x;
    confidence = static_cast<float>(maxScore);
    return predictedDigit >= 0 && predictedDigit <= 9;
}

bool isConfidentKnnPrediction(float prediction, const cv::Mat& neighborResponses, const cv::Mat& dists) {
    const int predictedDigit = static_cast<int>(std::round(prediction));
    if (predictedDigit < 1 || predictedDigit > 9 || neighborResponses.empty()) {
        return false;
    }

    int matchingNeighbors = 0;
    for (int i = 0; i < neighborResponses.cols; ++i) {
        const int neighborDigit = static_cast<int>(std::round(neighborResponses.at<float>(0, i)));
        if (neighborDigit == predictedDigit) {
            ++matchingNeighbors;
        }
    }

    if (matchingNeighbors < 2) {
        return false;
    }

    if (dists.cols >= 2) {
        float closestMatchingDist = std::numeric_limits<float>::max();
        float closestDifferentDist = std::numeric_limits<float>::max();

        for (int i = 0; i < dists.cols; ++i) {
            const int neighborDigit = static_cast<int>(std::round(neighborResponses.at<float>(0, i)));
            const float dist = dists.at<float>(0, i);
            if (neighborDigit == predictedDigit) {
                closestMatchingDist = std::min(closestMatchingDist, dist);
            } else {
                closestDifferentDist = std::min(closestDifferentDist, dist);
            }
        }

        if (closestDifferentDist < std::numeric_limits<float>::max()) {
            if (closestMatchingDist > closestDifferentDist * (1.0F - kAmbiguityGapMin)) {
                return false;
            }
        }
    }

    return true;
}

DigitRecognitionMetrics buildRecognitionMetrics(float prediction,
                                                const cv::Mat& neighborResponses,
                                                const cv::Mat& dists) {
    DigitRecognitionMetrics metrics;
    metrics.predictedDigit = static_cast<int>(std::round(prediction));

    if (neighborResponses.empty() || metrics.predictedDigit < 1 || metrics.predictedDigit > 9) {
        metrics.predictedDigit = 0;
        metrics.lowConfidence = true;
        return metrics;
    }

    for (int i = 0; i < dists.cols; ++i) {
        metrics.neighborDistances.push_back(dists.at<float>(0, i));
    }

    for (int i = 0; i < neighborResponses.cols; ++i) {
        const int neighborDigit = static_cast<int>(std::round(neighborResponses.at<float>(0, i)));
        if (neighborDigit == metrics.predictedDigit) {
            ++metrics.agreeingNeighbors;
        }
    }

    float averageDistance = 0.0F;
    float maxDistance = 0.0F;
    if (!metrics.neighborDistances.empty()) {
        for (float dist : metrics.neighborDistances) {
            averageDistance += dist;
            maxDistance = std::max(maxDistance, dist);
        }
        averageDistance /= static_cast<float>(metrics.neighborDistances.size());
    }

    const bool highDisagreement = metrics.agreeingNeighbors < kMinAgreeingNeighborsForHighConfidence;
    const bool largeDistances = averageDistance > kLowConfidenceAverageDistanceThreshold ||
                                maxDistance > kLowConfidenceMaxDistanceThreshold;

    metrics.lowConfidence = highDisagreement || largeDistances;
    return metrics;
}

cv::Ptr<cv::ml::KNearest> createDigitKnn() {
    namespace fs = std::filesystem;

    const fs::path templatesDir = fs::path("assets") / "templates";
    cv::Mat trainingSamples;
    cv::Mat trainingLabels;

    const cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

    for (int digit = 1; digit <= 9; ++digit) {
        const fs::path templatePath = templatesDir / (std::to_string(digit) + ".jpg");
        const cv::Mat rawTemplate = cv::imread(templatePath.string(), cv::IMREAD_GRAYSCALE);
        if (rawTemplate.empty()) {
            return nullptr;
        }

        cv::Mat normalizedTemplate = prepareTemplateForMatching(rawTemplate);
        if (normalizedTemplate.empty()) {
            return nullptr;
        }

        std::vector<cv::Mat> variants;
        variants.push_back(normalizedTemplate);

        cv::Mat eroded;
        cv::erode(normalizedTemplate, eroded, morphKernel, cv::Point(-1, -1), 1);
        variants.push_back(eroded);

        cv::Mat dilated;
        cv::dilate(normalizedTemplate, dilated, morphKernel, cv::Point(-1, -1), 1);
        variants.push_back(dilated);

        for (const cv::Mat& variant : variants) {
            trainingSamples.push_back(flattenSample(variant));
            trainingLabels.push_back(static_cast<float>(digit));
        }
    }

    if (trainingSamples.rows < 9 || trainingSamples.cols != (kNormalizedDigitSize * kNormalizedDigitSize)) {
        return nullptr;
    }

    cv::Mat labels32S;
    trainingLabels.convertTo(labels32S, CV_32S);

    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(3);
    knn->setIsClassifier(true);

    if (!knn->train(trainingSamples, cv::ml::ROW_SAMPLE, labels32S)) {
        return nullptr;
    }

    return knn;
}
}

cv::Mat ImageProcessor::extractSudokuGrid(const std::string& imagePath) {
    const cv::Mat loadedImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    return extractSudokuGrid(loadedImage);
}

cv::Mat ImageProcessor::extractSudokuGrid(const cv::Mat& inputImage) {
    originalImage_ = inputImage.clone();
    thresholdImage_.release();

    if (originalImage_.empty()) {
        return {};
    }

    BoardDetector boardDetector;
    cv::Mat warpedBoard;
    if (!boardDetector.detectBoard(originalImage_, warpedBoard)) {
        return {};
    }

    return warpedBoard;
}

std::vector<std::vector<cv::Mat>> ImageProcessor::splitIntoCells(const cv::Mat& warpedGrid) {
    if (warpedGrid.empty()) {
        return {};
    }

    const std::vector<cv::Mat> flatCells = splitSudokuBoardInto81Cells(warpedGrid);
    if (flatCells.size() != 81) {
        return {};
    }

    std::vector<std::vector<cv::Mat>> cells(9, std::vector<cv::Mat>(9));
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            cells[row][col] = flatCells[row * 9 + col];
        }
    }

    return cells;
}

bool ImageProcessor::detectDigit(const cv::Mat& cell, cv::Mat& normalizedDigit) const {
    normalizedDigit.release();
    if (cell.empty()) {
        return false;
    }

    try {
        cv::Mat gray;
        if (cell.channels() == 1) {
            gray = cell.clone();
        } else {
            cv::cvtColor(cell, gray, cv::COLOR_BGR2GRAY);
        }

        const int minSide = std::min(gray.rows, gray.cols);
        const int margin = std::max(1, minSide / 10);
        if (gray.cols <= margin * 2 || gray.rows <= margin * 2) {
            return false;
        }

        cv::Mat inner = gray(cv::Rect(margin, margin, gray.cols - margin * 2, gray.rows - margin * 2)).clone();

        cv::Mat blurred;
        cv::GaussianBlur(inner, blurred, cv::Size(3, 3), 0.0);

        cv::Mat thresholded;
        cv::adaptiveThreshold(
            blurred,
            thresholded,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            11,
            2
        );

        const int cellArea = thresholded.rows * thresholded.cols;
        if (cellArea <= 0) {
            return false;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresholded.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

#ifdef _DEBUG
        std::cout << "[DEBUG] detectDigit contour count: " << contours.size() << std::endl;
#endif

        const double minContourArea = static_cast<double>(cellArea) * 0.007;
        const int borderPadding = std::max(1, std::min(thresholded.rows, thresholded.cols) / 12);
        int largestContourIndex = -1;
        double largestContourArea = 0.0;

        for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
            const double area = cv::contourArea(contours[i]);
            if (area < minContourArea) {
                continue;
            }

            const cv::Rect rect = cv::boundingRect(contours[i]);
            const bool touchesBorder =
                rect.x <= borderPadding ||
                rect.y <= borderPadding ||
                (rect.x + rect.width) >= (thresholded.cols - borderPadding) ||
                (rect.y + rect.height) >= (thresholded.rows - borderPadding);

            if (touchesBorder) {
                const double aspect = static_cast<double>(rect.width) / static_cast<double>(rect.height);
                const bool likelyGridLine = aspect > 5.0 || aspect < 0.2;
                if (likelyGridLine) {
                    continue;
                }
            }

            if (area > largestContourArea) {
                largestContourArea = area;
                largestContourIndex = i;
            }
        }

#ifdef _DEBUG
        std::cout << "[DEBUG] detectDigit largest contour area: " << largestContourArea << std::endl;
#endif

        if (largestContourIndex < 0) {
            return false;
        }

        const cv::Rect bbox = cv::boundingRect(contours[largestContourIndex]);
        const int bboxArea = bbox.width * bbox.height;

#ifdef _DEBUG
        std::cout << "[DEBUG] detectDigit bounding box size: "
                  << bbox.width << "x" << bbox.height << std::endl;
#endif

        if (bboxArea < static_cast<int>(std::round(static_cast<double>(cellArea) * 0.015))) {
            return false;
        }

        cv::Mat largestMask(thresholded.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(largestMask, contours, largestContourIndex, cv::Scalar(255), cv::FILLED);

        cv::Mat isolated;
        cv::bitwise_and(thresholded, largestMask, isolated);

        normalizedDigit = extractDigitFromSudokuCell(isolated(bbox));
        const bool hasDigit = !normalizedDigit.empty() && cv::countNonZero(normalizedDigit) >= kMinNormalizedDigitInk;
        return hasDigit;
    } catch (const cv::Exception&) {
        normalizedDigit.release();
        return false;
    }
}

int ImageProcessor::detectDigit(const cv::Mat& cell) const {
    cv::Mat normalizedDigit;
    return detectDigit(cell, normalizedDigit) ? 1 : 0;
}

DigitRecognitionMetrics ImageProcessor::recognizeDigitWithMetrics(const cv::Mat& cell) {
    DigitRecognitionMetrics metrics;

    if (cell.empty()) {
        return metrics;
    }

    try {
        static cv::Ptr<cv::ml::KNearest> knn = createDigitKnn();
        if (!knn) {
            return metrics;
        }

        cv::Mat gray;
        if (cell.channels() == 1) {
            gray = cell.clone();
        } else {
            cv::cvtColor(cell, gray, cv::COLOR_BGR2GRAY);
        }

        cv::Mat normalizedDigit;
        if (!extractDigitMask(gray, normalizedDigit) || normalizedDigit.empty()) {
            return metrics;
        }

        if (cv::countNonZero(normalizedDigit) < kMinNormalizedDigitInk) {
            return metrics;
        }

        int cnnDigit = 0;
        float cnnConfidence = 0.0F;
        if (predictDigitWithCnn(normalizedDigit, cnnDigit, cnnConfidence) &&
            cnnDigit >= 1 && cnnDigit <= 9 &&
            cnnConfidence >= kCnnPredictionConfidenceThreshold) {
            metrics.predictedDigit = cnnDigit;
            metrics.lowConfidence = false;
            return metrics;
        }

        const cv::Mat sample = flattenSample(normalizedDigit);

        cv::Mat results;
        cv::Mat neighborResponses;
        cv::Mat dists;
        const float prediction = knn->findNearest(sample, 3, results, neighborResponses, dists);

        metrics = buildRecognitionMetrics(prediction, neighborResponses, dists);

        const bool knnIsConfident = isConfidentKnnPrediction(prediction, neighborResponses, dists);
        if (!knnIsConfident) {
            loadTemplates();
            if (digitTemplates_.size() == 9) {
                const TemplateMatchResult templateMatch = matchAgainstTemplates(normalizedDigit, digitTemplates_);
                const bool templateConfident =
                    templateMatch.digit >= 1 && templateMatch.digit <= 9 &&
                    templateMatch.bestScore >= kTemplateConfidenceMin &&
                    (templateMatch.bestScore - templateMatch.secondBestScore) >= kAmbiguityGapMin;

                if (templateConfident) {
                    metrics.predictedDigit = templateMatch.digit;
                    metrics.lowConfidence = false;
                    return metrics;
                }
            }

            metrics.predictedDigit = 0;
            metrics.lowConfidence = true;
        }

#ifdef _DEBUG
        std::cout << "[KNN] cell(" << gCurrentDebugRow << "," << gCurrentDebugCol << ")"
                  << " predicted=" << metrics.predictedDigit
                  << " agreeing=" << metrics.agreeingNeighbors
                  << " dists=[";
        for (size_t i = 0; i < metrics.neighborDistances.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << metrics.neighborDistances[i];
        }
        std::cout << "] lowConfidence=" << (metrics.lowConfidence ? "true" : "false") << std::endl;
#endif

        return metrics;
    } catch (const cv::Exception&) {
        return DigitRecognitionMetrics{};
    }
}

int ImageProcessor::recognizeDigit(const cv::Mat& cell) {
    return recognizeDigitWithMetrics(cell).predictedDigit;
}

std::vector<std::vector<int>> ImageProcessor::extractDigitMatrix(const std::vector<std::vector<cv::Mat>>& cells) {
    if (cells.size() != 9) {
        return {};
    }

    std::vector<std::vector<int>> matrix(9, std::vector<int>(9, 0));
    lastRecognitionMetrics_ = std::vector<std::vector<DigitRecognitionMetrics>>(9, std::vector<DigitRecognitionMetrics>(9));
    int recognizedCells = 0;

    for (int row = 0; row < 9; ++row) {
        if (cells[row].size() != 9) {
            return {};
        }

        for (int col = 0; col < 9; ++col) {
            const cv::Mat& cell = cells[row][col];
            if (cell.empty()) {
                matrix[row][col] = 0;
                lastRecognitionMetrics_[row][col] = DigitRecognitionMetrics{};
                continue;
            }

            try {
                const int margin = std::max(1, std::min(cell.rows, cell.cols) / 12);
                cv::Mat innerCell = cell;
                if (cell.cols > 2 * margin && cell.rows > 2 * margin) {
                    innerCell = cell(cv::Rect(
                        margin,
                        margin,
                        cell.cols - 2 * margin,
                        cell.rows - 2 * margin
                    ));
                }

                cv::Mat resolutionNormalizedCell = normalizeCellResolution(innerCell, row, col);
                if (resolutionNormalizedCell.empty()) {
                    matrix[row][col] = 0;
                    lastRecognitionMetrics_[row][col] = DigitRecognitionMetrics{};
                    continue;
                }

#ifdef _DEBUG
                setCurrentDebugCell(row, col);
#endif
                const DigitRecognitionMetrics metrics = recognizeDigitWithMetrics(resolutionNormalizedCell);
                lastRecognitionMetrics_[row][col] = metrics;
                matrix[row][col] = metrics.predictedDigit;
#ifdef _DEBUG
                setCurrentDebugCell(-1, -1);
#endif
                if (matrix[row][col] != 0) {
                    ++recognizedCells;
                }
            } catch (const cv::Exception&) {
#ifdef _DEBUG
                setCurrentDebugCell(-1, -1);
#endif
                matrix[row][col] = 0;
                lastRecognitionMetrics_[row][col] = DigitRecognitionMetrics{};
            }
        }
    }

    std::cout << "Recognized non-empty cells: " << recognizedCells << std::endl;

    std::vector<int> flatDigits;
    flatDigits.reserve(81);
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            flatDigits.push_back(matrix[row][col]);
        }
    }

    int sudokuDebug[9][9] = {};
    convertRecognizedDigitsToMatrix(flatDigits, sudokuDebug);

    return matrix;
}

const std::vector<std::vector<DigitRecognitionMetrics>>& ImageProcessor::getLastRecognitionMetrics() const {
    return lastRecognitionMetrics_;
}

void ImageProcessor::loadTemplates() {
    if (templatesLoaded_ && digitTemplates_.size() == 9) {
        return;
    }

    namespace fs = std::filesystem;
    const fs::path templatesDir = fs::path("assets") / "templates";

    std::vector<cv::Mat> loadedTemplates;
    loadedTemplates.reserve(9);

    for (int digit = 1; digit <= 9; ++digit) {
        const fs::path filePath = templatesDir / (std::to_string(digit) + ".jpg");
        cv::Mat templateImage = cv::imread(filePath.string(), cv::IMREAD_GRAYSCALE);
        if (templateImage.empty()) {
            templatesLoaded_ = false;
            digitTemplates_.clear();
            return;
        }

        cv::Mat thresholdedTemplate;
        cv::adaptiveThreshold(
            templateImage,
            thresholdedTemplate,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY,
            11,
            2
        );

        cv::Mat resizedTemplate;
        cv::resize(thresholdedTemplate, resizedTemplate, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
        loadedTemplates.push_back(resizedTemplate);
    }

    digitTemplates_ = loadedTemplates;
    templatesLoaded_ = (digitTemplates_.size() == 9);
}

const cv::Mat& ImageProcessor::getOriginalImage() const {
    return originalImage_;
}

const cv::Mat& ImageProcessor::getThresholdImage() const {
    return thresholdImage_;
}

bool ImageProcessor::findLargestQuadrilateral(const std::vector<std::vector<cv::Point>>& contours,
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

std::vector<cv::Point2f> ImageProcessor::orderCorners(const std::vector<cv::Point>& corners) const {
    if (corners.size() != 4) {
        return {};
    }

    std::vector<cv::Point2f> ordered(4);

    float minSum = std::numeric_limits<float>::max();
    float maxSum = std::numeric_limits<float>::lowest();
    float minDiff = std::numeric_limits<float>::max();
    float maxDiff = std::numeric_limits<float>::lowest();

    for (const auto& point : corners) {
        const float x = static_cast<float>(point.x);
        const float y = static_cast<float>(point.y);
        const float sum = x + y;
        const float diff = y - x;

        if (sum < minSum) {
            minSum = sum;
            ordered[0] = {x, y};
        }
        if (sum > maxSum) {
            maxSum = sum;
            ordered[2] = {x, y};
        }
        if (diff < minDiff) {
            minDiff = diff;
            ordered[1] = {x, y};
        }
        if (diff > maxDiff) {
            maxDiff = diff;
            ordered[3] = {x, y};
        }
    }

    return ordered;
}
