#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat clean_local_laplacian_filter(const cv::Mat &input) {
    int levels = 8;
    float alpha = 1.0f;
    float beta = 1.0f;
    cv::Mat gray, output = input.clone();
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> pyramid(levels);
    pyramid[0] = gray.clone();
    for (int i = 1; i < levels; ++i) {
        cv::pyrDown(pyramid[i - 1], pyramid[i]);
    }

    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            float gray_val = gray.at<float>(y, x);
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);

            for (int l = 0; l < levels; ++l) {
                float base = pyramid[l].at<float>(y >> l, x >> l) * alpha;
                float diff = std::clamp(gray_val - base, -beta, beta);
                float weight = std::pow(2.0f, -l);

                for (int c = 0; c < 3; ++c) {
                    output.at<cv::Vec3b>(y, x)[c] = std::clamp(pixel[c] + diff * weight * 255.0f, 0.0f, 255.0f);
                }
            }
        }
    }
    return output;
}
