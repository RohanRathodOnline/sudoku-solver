#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "httplib.h"
#include "image_processor.h"
#include "sudoku_solver.h"

void startServer();

namespace {

    void printSudokuGrid(const std::vector<std::vector<int>>& grid, const std::string& title) {
        std::cout << "----------------------------------------\n" << title << "\n";
        for (int row = 0; row < 9; ++row) {
            if (row > 0 && row % 3 == 0) { std::cout << "------+-------+------\n"; }
            for (int col = 0; col < 9; ++col) {
                if (col > 0 && col % 3 == 0) { std::cout << "| "; }
                std::cout << grid[row][col] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << "----------------------------------------\n";
    }

    void generateTemplates() {
        namespace fs = std::filesystem;
        const fs::path templatesDir = fs::path("assets") / "templates";
        fs::create_directories(templatesDir);
        constexpr int    canvasSize = 100;
        constexpr int    outputSize = 28;
        constexpr double fontScale = 2.4;
        constexpr int    thickness = 4;
        for (int digit = 1; digit <= 9; ++digit) {
            cv::Mat canvas(canvasSize, canvasSize, CV_8UC1, cv::Scalar(255));
            int baseline = 0;
            const std::string text = std::to_string(digit);
            const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
            cv::putText(canvas, text,
                cv::Point((canvas.cols - textSize.width) / 2,
                    (canvas.rows + textSize.height) / 2),
                cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(0), thickness, cv::LINE_AA);
            cv::Mat output;
            cv::resize(canvas, output, cv::Size(outputSize, outputSize), 0, 0, cv::INTER_AREA);
            cv::imwrite((templatesDir / (text + ".jpg")).string(), output);
        }
        std::cout << "Templates generated in " << templatesDir.string() << "\n";
    }

} // namespace

int main(int argc, char* argv[]) {

    // generateTemplates(); // run once to generate templates, then comment out

    if (argc >= 2) {
        const std::string imagePath = argv[1];
        const cv::Mat inputImage = cv::imread(imagePath);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load image from: " << imagePath << "\n";
            return 1;
        }
        std::cout << "Image loaded: " << imagePath << "\n";

        ImageProcessor processor;
        cv::Mat warpedGrid = processor.extractSudokuGrid(imagePath);
        if (warpedGrid.empty()) {
            std::cerr << "Error: Failed to detect Sudoku grid.\n";
            return 1;
        }

        const auto cells = processor.splitIntoCells(warpedGrid);
        auto       matrix = processor.extractDigitMatrix(cells);

        printSudokuGrid(matrix, "Detected Sudoku:");

        SudokuSolver solver;
        if (solver.solve(matrix)) {
            printSudokuGrid(matrix, "Solved Sudoku:");
        }
        else {
            std::cout << "Could not solve — some digits may have been misread.\n";
        }

        cv::imshow("Warped Grid", warpedGrid);
        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
    }

    startServer();
    return 0;
}