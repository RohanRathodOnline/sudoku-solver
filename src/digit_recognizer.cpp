#include "digit_recognizer.h"

#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
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
}

DigitRecognizer::DigitRecognizer(const std::string& modelPath) {
    const std::filesystem::path resolvedModelPath = resolveModelPath(modelPath);
    std::cout << "Loading CNN model from: " << resolvedModelPath.string() << std::endl;

    if (!std::filesystem::exists(resolvedModelPath)) {
        std::cerr << "ERROR: CNN model not found at " << resolvedModelPath.string() << std::endl;
        std::cerr << "Ensure assets/models/mnist.onnx exists in the runtime directory." << std::endl;
        loaded_ = false;
        return;
    }

    try {
        net_ = cv::dnn::readNet(resolvedModelPath.string());
        loaded_ = !net_.empty();
        if (loaded_) {
            std::cout << "CNN model loaded successfully." << std::endl;
        } else {
            std::cerr << "ERROR: Failed to load CNN model from " << resolvedModelPath.string() << std::endl;
        }
    } catch (const cv::Exception&) {
        std::cerr << "ERROR: OpenCV failed to read CNN model at " << resolvedModelPath.string() << std::endl;
        loaded_ = false;
    }
}

int DigitRecognizer::recognizeDigit(const cv::Mat& digit) {
    if (!loaded_ || digit.empty()) {
        return -1;
    }

    cv::Mat gray;
    if (digit.channels() == 1) {
        gray = digit;
    } else {
        cv::cvtColor(digit, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(28, 28), 0.0, 0.0, cv::INTER_AREA);

    cv::Mat blob = cv::dnn::blobFromImage(
        resized,
        1.0 / 255.0,
        cv::Size(28, 28),
        cv::Scalar(),
        false,
        false,
        CV_32F
    );

    net_.setInput(blob);
    cv::Mat output = net_.forward();
    if (output.empty()) {
        return -1;
    }

    cv::Mat scores = output.reshape(1, 1);
    cv::Point classIdPoint;
    double maxScore = 0.0;
    cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);

    return classIdPoint.x;
}

bool DigitRecognizer::isLoaded() const {
    return loaded_;
}
