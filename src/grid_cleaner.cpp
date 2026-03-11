#include "grid_cleaner.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

cv::Mat GridCleaner::removeGridLines(const cv::Mat& board) const {
    if (board.empty()) {
        return {};
    }

    cv::Mat gray;
    if (board.channels() == 1) {
        gray = board.clone();
    } else {
        cv::cvtColor(board, gray, cv::COLOR_BGR2GRAY);
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

    const int rows = std::max(1, binary.rows);
    const int cols = std::max(1, binary.cols);

    // FIX: Use 2/3 of the board dimension so only lines that span multiple
    // cells are removed. The old rows/20 (≈22px) was too small — it wiped
    // out thin digit strokes like 1, 7 and 4 as well as grid lines.
    // Grid lines cross the entire board; digits never exceed one cell (~50px).
    const int vertLen   = std::max(1, rows * 2 / 3);
    const int horizLen  = std::max(1, cols * 2 / 3);

    cv::Mat verticalMask;
    cv::Mat horizontalMask;

    const cv::Mat verticalKernel   = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, vertLen));
    const cv::Mat horizontalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizLen, 1));

    cv::erode(binary,         verticalMask,   verticalKernel);
    cv::dilate(verticalMask,  verticalMask,   verticalKernel);

    cv::erode(binary,          horizontalMask, horizontalKernel);
    cv::dilate(horizontalMask, horizontalMask, horizontalKernel);

    cv::Mat gridMask;
    cv::bitwise_or(verticalMask, horizontalMask, gridMask);

    cv::Mat cleaned;
    cv::subtract(binary, gridMask, cleaned);

    return cleaned;
}
