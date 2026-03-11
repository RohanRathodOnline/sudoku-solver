#include "cell_extractor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace {
constexpr int kWarpedBoardSize = 450;
}

std::vector<cv::Mat> CellExtractor::extractCells(const cv::Mat& board) const {
    if (board.empty()) {
        return {};
    }

    cv::Mat normalizedBoard;
    if (board.rows != kWarpedBoardSize || board.cols != kWarpedBoardSize) {
        cv::resize(board, normalizedBoard, cv::Size(kWarpedBoardSize, kWarpedBoardSize), 0.0, 0.0, cv::INTER_AREA);
    } else {
        normalizedBoard = board;
    }

    const int cellSize = kWarpedBoardSize / 9;
    std::vector<cv::Mat> cells;
    cells.reserve(81);

    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            const int x = col * cellSize;
            const int y = row * cellSize;
            const int width = (col == 8) ? (kWarpedBoardSize - x) : cellSize;
            const int height = (row == 8) ? (kWarpedBoardSize - y) : cellSize;

            cells.push_back(normalizedBoard(cv::Rect(x, y, width, height)).clone());
        }
    }

    return cells;
}
