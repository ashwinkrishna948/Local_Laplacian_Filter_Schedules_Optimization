#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include "clean_local_laplacian.cpp"
#include "fast_local_laplacian.cpp"
#include "halide_local_laplacian.cpp"

void benchmark(const std::string &name, const cv::Mat &input, cv::Mat(*filter)(const cv::Mat &)) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat output = filter(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " took " << duration.count() << " seconds" << std::endl;
}

int main() {
    cv::Mat input = cv::imread("sample_image.jpg");
    if (input.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    benchmark("Clean C++", input, clean_local_laplacian_filter);
    benchmark("Fast C++", input, fast_local_laplacian_filter);
    benchmark("Halide", input, halide_local_laplacian_filter);

    return 0;
}
