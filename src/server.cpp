#include "httplib.h"
#include "image_intake_service.h"
#include "sudoku_solver.h"
#include "vision_pipeline.h"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>  // ADD THIS LINE

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
struct SolveSudokuResult {
    std::vector<std::vector<int>> rawDetected;
    std::vector<std::vector<int>> detected;
    std::vector<std::vector<int>> solved;
    bool solvedSuccessfully = false;
    int removedConflicts = 0;
    int fallbackDropped = 0;
};

bool hasGivenConflict(const std::vector<std::vector<int>>& matrix, int row, int col);

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

void printCnnModelStartupDiagnostics() {
    const std::filesystem::path modelPath = resolveModelPath("assets/models/mnist.onnx");
    std::cout << "Loading CNN model from: " << modelPath.string() << std::endl;

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "ERROR: CNN model not found at " << modelPath.string() << std::endl;
        std::cerr << "Ensure assets/models/mnist.onnx exists in the runtime directory." << std::endl;
        return;
    }

    try {
        cv::dnn::Net net = cv::dnn::readNet(modelPath.string());
        if (net.empty()) {
            std::cerr << "ERROR: Failed to load CNN model from " << modelPath.string() << std::endl;
            return;
        }

        std::cout << "CNN model loaded successfully." << std::endl;
    } catch (const cv::Exception&) {
        std::cerr << "ERROR: OpenCV failed to read CNN model at " << modelPath.string() << std::endl;
    }
}

std::string matrixToJson(const std::vector<std::vector<int>>& matrix) {
    std::ostringstream out;
    out << "[";

    for (size_t row = 0; row < matrix.size(); ++row) {
        if (row > 0) {
            out << ",";
        }

        out << "[";
        for (size_t col = 0; col < matrix[row].size(); ++col) {
            if (col > 0) {
                out << ",";
            }
            out << matrix[row][col];
        }
        out << "]";
    }

    out << "]";
    return out.str();
}

std::string buildImageIntakeErrorJson(const std::string& code, const std::string& message) {
    std::ostringstream out;
    out << "{\"status\":\"error\",\"error\":{\"code\":\"" << code
        << "\",\"message\":\"" << message << "\"}}";
    return out.str();
}

std::string buildStructuredErrorJson(const std::string& code, const std::string& message) {
    std::ostringstream out;
    out << "{\"status\":\"error\",\"error\":{\"code\":\"" << code
        << "\",\"message\":\"" << message << "\"}}";
    return out.str();
}

int mapImageIntakeErrorStatus(const std::string& code) {
    if (code == "E_UPLOAD_EMPTY") {
        return 400;
    }

    if (code == "E_IMAGE_TOO_LARGE") {
        return 413;
    }

    if (code == "E_DECODE_FAILED") {
        return 422;
    }

    return 400;
}

bool parseManualBoardFromJson(const std::string& body, std::vector<std::vector<int>>& board) {
    std::vector<int> values;
    values.reserve(81);

    int current = -1;
    for (char ch : body) {
        if (ch >= '0' && ch <= '9') {
            if (current < 0) {
                current = 0;
            }
            current = current * 10 + (ch - '0');
        } else if (current >= 0) {
            values.push_back(current);
            current = -1;
        }
    }
    if (current >= 0) {
        values.push_back(current);
    }

    if (values.size() != 81) {
        return false;
    }

    board.assign(9, std::vector<int>(9, 0));
    for (size_t i = 0; i < values.size(); ++i) {
        const int value = values[i];
        if (value < 0 || value > 9) {
            return false;
        }
        board[i / 9][i % 9] = value;
    }

    return true;
}

std::string buildErrorJson(const std::string& message,
                           const std::vector<std::vector<int>>* detected = nullptr,
                           const std::vector<std::vector<int>>* rawDetected = nullptr,
                           int removedConflicts = 0,
                           int fallbackDropped = 0) {
    std::ostringstream out;
    out << "{\"error\":\"" << message << "\"";
    if (detected && detected->size() == 9) {
        out << ",\"detected\":" << matrixToJson(*detected);
        out << ",\"cleanedDetected\":" << matrixToJson(*detected);
    }
    if (rawDetected && rawDetected->size() == 9) {
        out << ",\"rawDetected\":" << matrixToJson(*rawDetected);
    }
    out << ",\"meta\":{\"removedConflicts\":" << removedConflicts
        << ",\"fallbackDropped\":" << fallbackDropped << "}";
    out << "}";
    return out.str();
}

bool hasAnyBoardConflict(const std::vector<std::vector<int>>& matrix) {
    if (matrix.size() != 9) {
        return true;
    }

    for (int row = 0; row < 9; ++row) {
        if (matrix[row].size() != 9) {
            return true;
        }

        for (int col = 0; col < 9; ++col) {
            if (matrix[row][col] == 0) {
                continue;
            }

            if (hasGivenConflict(matrix, row, col)) {
                return true;
            }
        }
    }

    return false;
}

int countFilledCells(const std::vector<std::vector<int>>& matrix) {
    int filled = 0;
    for (const auto& row : matrix) {
        for (int value : row) {
            if (value != 0) {
                ++filled;
            }
        }
    }
    return filled;
}

bool hasGivenConflict(const std::vector<std::vector<int>>& matrix, int row, int col) {
    if (matrix.size() != 9 || matrix[row].size() != 9) {
        return true;
    }

    const int value = matrix[row][col];
    if (value == 0) {
        return false;
    }

    for (int c = 0; c < 9; ++c) {
        if (c != col && matrix[row][c] == value) {
            return true;
        }
    }

    for (int r = 0; r < 9; ++r) {
        if (r != row && matrix[r][col] == value) {
            return true;
        }
    }

    const int boxStartRow = (row / 3) * 3;
    const int boxStartCol = (col / 3) * 3;
    for (int r = boxStartRow; r < boxStartRow + 3; ++r) {
        for (int c = boxStartCol; c < boxStartCol + 3; ++c) {
            if ((r != row || c != col) && matrix[r][c] == value) {
                return true;
            }
        }
    }

    return false;
}

int removeConflictingGivens(std::vector<std::vector<int>>& matrix) {
    if (matrix.size() != 9) {
        return 0;
    }

    int removed = 0;
    bool changed = false;

    do {
        changed = false;
        for (int row = 0; row < 9; ++row) {
            if (matrix[row].size() != 9) {
                continue;
            }

            for (int col = 0; col < 9; ++col) {
                if (matrix[row][col] == 0) {
                    continue;
                }

                if (hasGivenConflict(matrix, row, col)) {
                    matrix[row][col] = 0;
                    ++removed;
                    changed = true;
                }
            }
        }
    } while (changed);

    return removed;
}

bool trySolveByDroppingGivens(const std::vector<std::vector<int>>& baseMatrix,
                              std::vector<std::vector<int>>& solvedMatrix,
                              int& droppedCount) {
    droppedCount = 0;
    if (baseMatrix.size() != 9) {
        return false;
    }

    std::vector<std::pair<int, int>> givens;
    for (int row = 0; row < 9; ++row) {
        if (baseMatrix[row].size() != 9) {
            return false;
        }

        for (int col = 0; col < 9; ++col) {
            if (baseMatrix[row][col] != 0) {
                givens.emplace_back(row, col);
            }
        }
    }

    SudokuSolver solver;

    for (size_t i = 0; i < givens.size(); ++i) {
        auto trial = baseMatrix;
        trial[givens[i].first][givens[i].second] = 0;
        if (solver.solve(trial)) {
            solvedMatrix = std::move(trial);
            droppedCount = 1;
            return true;
        }
    }

    for (size_t i = 0; i < givens.size(); ++i) {
        for (size_t j = i + 1; j < givens.size(); ++j) {
            auto trial = baseMatrix;
            trial[givens[i].first][givens[i].second] = 0;
            trial[givens[j].first][givens[j].second] = 0;

            if (solver.solve(trial)) {
                solvedMatrix = std::move(trial);
                droppedCount = 2;
                return true;
            }
        }
    }

    return false;
}

cv::Mat normalizeDigitToCanvas(const cv::Mat& digitImage) {
    if (digitImage.empty()) {
        return cv::Mat();
    }

    const int targetSize = 28;
    const int padding = 4;
    const int digitSize = targetSize - 2 * padding;

    cv::Mat resized;
    cv::resize(digitImage, resized, cv::Size(digitSize, digitSize), 0, 0, cv::INTER_AREA);

    cv::Mat canvas = cv::Mat::zeros(targetSize, targetSize, CV_8UC1);
    resized.copyTo(canvas(cv::Rect(padding, padding, digitSize, digitSize)));

    return canvas;
}

} // end namespace

SolveSudokuResult solveSudokuFromImage(const cv::Mat& decodedImage) {
    VisionPipeline pipeline;
    const PipelineResult pipelineResult = pipeline.process(decodedImage);

    if (!pipelineResult.success) {
        return {
            pipelineResult.detectedGrid,
            pipelineResult.detectedGrid,
            {},
            false,
            0,
            0
        };
    }

    return {
        pipelineResult.detectedGrid,
        pipelineResult.detectedGrid,
        pipelineResult.solvedGrid,
        true,
        0,
        0
    };
}

void startServer() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    printCnnModelStartupDiagnostics();

    httplib::Server server;

    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });

    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    server.Post("/solve-manual", [](const httplib::Request& req, httplib::Response& res) {
        std::vector<std::vector<int>> board;
        if (!parseManualBoardFromJson(req.body, board)) {
            res.status = 400;
            res.set_content(
                buildStructuredErrorJson("E_INVALID_BOARD_PAYLOAD", "Request body must include a 9x9 board with values 0-9"),
                "application/json"
            );
            return;
        }

        if (hasAnyBoardConflict(board)) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson("E_INVALID_BOARD", "Manual board contains conflicting givens"),
                "application/json"
            );
            return;
        }

        auto solved = board;
        SudokuSolver solver;
        if (!solver.solve(solved)) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson("E_SOLVER_FAILED", "Sudoku solver failed to produce a valid solution"),
                "application/json"
            );
            return;
        }

        std::ostringstream out;
        out << "{\"detected\":" << matrixToJson(board)
            << ",\"solved\":" << matrixToJson(solved) << "}";
        res.set_content(out.str(), "application/json");
    });

    server.Post("/solve-image", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data() || !req.form.has_file("image")) {
            res.status = 400;
            res.set_content(
                buildStructuredErrorJson("E_GRID_NOT_FOUND", "Missing form-data field 'image'"),
                "application/json"
            );
            return;
        }

        const auto file = req.form.get_file("image");
        ImageIntakeService imageIntakeService;
        const ImageIntakeResult intakeResult = imageIntakeService.decodeAndPrepare(file.content);
        if (!intakeResult.success) {
            res.status = mapImageIntakeErrorStatus(intakeResult.errorCode);
            res.set_content(
                buildImageIntakeErrorJson(intakeResult.errorCode, intakeResult.errorMessage),
                "application/json"
            );
            return;
        }

        const auto result = solveSudokuFromImage(intakeResult.image);

        std::cout << "\n===== DETECTED MATRIX =====\n";
        for (const auto& row : result.rawDetected) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "===========================\n";

        std::cout << "\n===== CLEANED MATRIX =====\n";
        for (const auto& row : result.detected) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "==========================\n";

        if (result.rawDetected.size() != 9) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson("E_GRID_NOT_FOUND", "Sudoku grid could not be extracted from image"),
                "application/json"
            );
            return;
        }

        if (countFilledCells(result.rawDetected) == 0) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson("E_EMPTY_DETECTION", "No digits detected in extracted Sudoku grid"),
                "application/json"
            );
            return;
        }

        if (countFilledCells(result.detected) < 17) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson(
                    "E_DIGIT_RECOGNITION_FAILED",
                    "Digit recognition returned insufficient valid cells"
                ),
                "application/json"
            );
#ifdef _DEBUG
            std::cout << "[ERROR] 422 Triggered - DIGIT_RECOGNITION_FAILED" << std::endl;
#endif
            return;
        }

        if (hasAnyBoardConflict(result.detected)) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson("E_INVALID_BOARD", "Detected board contains conflicting givens"),
                "application/json"
            );
            return;
        }

        if (!result.solvedSuccessfully) {
            res.status = 422;
            res.set_content(
                buildStructuredErrorJson("E_SOLVER_FAILED", "Sudoku solver failed to produce a valid solution"),
                "application/json"
            );
            return;
        }

        std::ostringstream json;
        json << "{\"rawDetected\":" << matrixToJson(result.rawDetected)
             << ",\"detected\":" << matrixToJson(result.detected)
             << ",\"cleanedDetected\":" << matrixToJson(result.detected)
             << ",\"solved\":" << matrixToJson(result.solved)
             << ",\"meta\":{\"removedConflicts\":" << result.removedConflicts
             << ",\"fallbackDropped\":" << result.fallbackDropped << "}}";

        res.set_content(json.str(), "application/json");
    });

    std::cout << "Server running at http://localhost:8080" << std::endl;
    server.listen("0.0.0.0", 8080);
}

