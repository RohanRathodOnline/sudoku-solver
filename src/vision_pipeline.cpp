#include "vision_pipeline.h"

#include "board_detector.h"
#include "cell_extractor.h"
#include "digit_recognizer.h"
#include "digit_segmenter.h"
#include "grid_cleaner.h"
#include "image_processor.h"
#include "sudoku_solver.h"

#include <chrono>
#include <iostream>

namespace {
using Clock = std::chrono::high_resolution_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

// FIX: Create DigitRecognizer ONCE here — loads mnist.onnx a single time.
// The old code declared these as local variables inside recognizeDigits(),
// so the ONNX model was read from disk on every single HTTP request.
VisionPipeline::VisionPipeline(const std::string& modelPath)
    : recognizer_(std::make_unique<DigitRecognizer>(modelPath))
    , fallbackRecognizer_(std::make_unique<ImageProcessor>())
{
    if (!recognizer_->isLoaded()) {
        std::cerr << "[VisionPipeline] WARNING: CNN model failed to load from \""
                  << modelPath << "\". Digit recognition will use KNN fallback only.\n";
    }
}

VisionPipeline::~VisionPipeline() = default;

PipelineResult VisionPipeline::process(const cv::Mat& image) {
    PipelineResult result;
    const auto totalStart = Clock::now();

    if (image.empty()) {
        result.errorMessage = "Input image is empty";
        return result;
    }

    // ── Stage 1: Board detection & perspective warp ──────────────────────────
    const auto boardStart = Clock::now();
    cv::Mat warpedGrid = detectBoardAndWarp(image);
    result.timings.boardDetectionMs  = elapsedMs(boardStart, Clock::now());
    result.timings.perspectiveWarpMs = result.timings.boardDetectionMs;

    if (warpedGrid.empty()) {
        result.errorMessage = "Sudoku board not found in image. "
                              "Make sure the whole grid is visible and the photo is sharp.";
        result.timings.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }

    // ── Stage 2: Grid line removal ───────────────────────────────────────────
    const auto removeLinesStart = Clock::now();
    cv::Mat gridWithoutLines = removeGridLines(warpedGrid);
    result.timings.gridLineRemovalMs = elapsedMs(removeLinesStart, Clock::now());

    if (gridWithoutLines.empty()) {
        result.errorMessage = "Grid line removal failed";
        result.timings.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }

    // ── Stage 3: Cell extraction ─────────────────────────────────────────────
    const auto cellsStart = Clock::now();
    const auto cells = extractCells(gridWithoutLines);
    result.timings.cellExtractionMs = elapsedMs(cellsStart, Clock::now());

    if (cells.size() != 9) {
        result.errorMessage = "Cell extraction failed — expected 9 rows, got "
                              + std::to_string(cells.size());
        result.timings.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }

    // ── Stage 4: Digit segmentation ──────────────────────────────────────────
    const auto segmentationStart = Clock::now();
    const auto segmentedDigits = segmentDigits(cells);
    result.timings.digitSegmentationMs = elapsedMs(segmentationStart, Clock::now());

    // ── Stage 5: Digit recognition ────────────────────────────────────────────
    const auto recognitionStart = Clock::now();
    result.detectedGrid = recognizeDigits(cells, segmentedDigits);
    result.timings.digitRecognitionMs = elapsedMs(recognitionStart, Clock::now());

    if (result.detectedGrid.size() != 9) {
        result.errorMessage = "Digit recognition failed";
        result.timings.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }

    // ── Stage 6: Solve ────────────────────────────────────────────────────────
    const auto solvingStart = Clock::now();
    const bool solved = solveGrid(result.detectedGrid, result.solvedGrid);
    result.timings.sudokuSolvingMs = elapsedMs(solvingStart, Clock::now());

    if (!solved) {
        result.errorMessage = "Sudoku solver could not find a valid solution. "
                              "Some digits may have been misread.";
        result.timings.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }

    result.success = true;
    result.timings.totalMs = elapsedMs(totalStart, Clock::now());
    return result;
}

cv::Mat VisionPipeline::detectBoardAndWarp(const cv::Mat& image) {
    BoardDetector detector;
    cv::Mat warpedBoard;
    if (!detector.detectBoard(image, warpedBoard)) {
        return {};
    }
    return warpedBoard;
}

cv::Mat VisionPipeline::removeGridLines(const cv::Mat& warpedGrid) const {
    GridCleaner cleaner;
    return cleaner.removeGridLines(warpedGrid);
}

std::vector<std::vector<cv::Mat>> VisionPipeline::extractCells(const cv::Mat& gridImage) {
    CellExtractor extractor;
    const std::vector<cv::Mat> flatCells = extractor.extractCells(gridImage);
    if (flatCells.size() != 81) {
        return {};
    }

    std::vector<std::vector<cv::Mat>> cells(9, std::vector<cv::Mat>(9));
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            cells[row][col] = flatCells[static_cast<size_t>(row * 9 + col)];
        }
    }
    return cells;
}

std::vector<std::vector<cv::Mat>> VisionPipeline::segmentDigits(
    const std::vector<std::vector<cv::Mat>>& cells)
{
    std::vector<std::vector<cv::Mat>> segmented(9, std::vector<cv::Mat>(9));
    DigitSegmenter segmenter;

    for (int row = 0; row < 9; ++row) {
        if (cells[row].size() != 9) {
            return {};
        }
        for (int col = 0; col < 9; ++col) {
            cv::Mat digitMask;
            if (segmenter.extractDigit(cells[row][col], digitMask)) {
                segmented[row][col] = digitMask;
            }
            // Empty cells leave segmented[row][col] as a default (empty) Mat.
        }
    }
    return segmented;
}

std::vector<std::vector<int>> VisionPipeline::recognizeDigits(
    const std::vector<std::vector<cv::Mat>>& cells,
    const std::vector<std::vector<cv::Mat>>& segmentedDigits)
{
    if (cells.size() != 9 || segmentedDigits.size() != 9) {
        return {};
    }

    std::vector<std::vector<int>> matrix(9, std::vector<int>(9, 0));

    for (int row = 0; row < 9; ++row) {
        if (cells[row].size() != 9 || segmentedDigits[row].size() != 9) {
            return {};
        }

        for (int col = 0; col < 9; ++col) {
            int predicted = 0;

            // Primary path: CNN (fast, loaded once at construction time)
            if (recognizer_->isLoaded() && !segmentedDigits[row][col].empty()) {
                predicted = recognizer_->recognizeDigit(segmentedDigits[row][col]);
            }

            // Fallback: KNN + template matching inside ImageProcessor
            // Only called when CNN is unavailable or returns an invalid digit.
            if (predicted < 1 || predicted > 9) {
                predicted = fallbackRecognizer_->recognizeDigit(cells[row][col]);
            }

            matrix[row][col] = (predicted >= 1 && predicted <= 9) ? predicted : 0;
        }
    }

    return matrix;
}

bool VisionPipeline::solveGrid(const std::vector<std::vector<int>>& detectedGrid,
                               std::vector<std::vector<int>>&       solvedGrid) const {
    solvedGrid = detectedGrid;
    SudokuSolver solver;
    return solver.solve(solvedGrid);
}
