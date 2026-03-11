cmake_minimum_required(VERSION 3.15)
project(SudokuBackendCMake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV CONFIG REQUIRED)

add_executable(SudokuBackendCMake
    src/main.cpp
    src/server.cpp
    src/sudoku_solver.cpp
    src/image_intake_service.cpp
    src/image_quality_analyzer.cpp
    src/image_processor.cpp
    src/board_detector.cpp
    src/grid_cleaner.cpp
    src/cell_extractor.cpp
    src/digit_segmenter.cpp
    src/digit_recognizer.cpp
    src/vision_pipeline.cpp
)

target_include_directories(SudokuBackendCMake
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(SudokuBackendCMake
    PRIVATE
        ${OpenCV_LIBS}
)

# Copy assets (including mnist.onnx model) to the build output directory
if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(
        TARGET SudokuBackendCMake POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:SudokuBackendCMake>/assets"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/assets"
                "$<TARGET_FILE_DIR:SudokuBackendCMake>/assets"
        COMMENT "Copying assets to output directory"
    )
endif()

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/assets/models/mnist.onnx")
    message(WARNING
        "CNN model NOT found at ${CMAKE_SOURCE_DIR}/assets/models/mnist.onnx. "
        "Copy mnist.onnx there before running — digit recognition will fail without it.")
endif()

