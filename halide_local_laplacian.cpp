#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace Halide;

cv::Mat halide_local_laplacian_filter(const cv::Mat &input) {
    Buffer<uint8_t> input_buf(input.data, input.cols, input.rows, 3);
    Var x, y, c;

    Func gray;
    gray(x, y) = cast<float>(0.299f * input_buf(x, y, 2) + 0.587f * input_buf(x, y, 1) + 0.114f * input_buf(x, y, 0)) / 255.0f;

    int levels = 8;
    float alpha = 1.0f, beta = 1.0f;
    Func output;
    Expr gray_val = gray(x, y);

    RDom r(0, levels);
    Expr base = gray(x >> r, y >> r) * alpha;
    Expr diff = clamp(gray_val - base, -beta, beta);
    Expr weight = pow(2.0f, -r);
    Expr color = clamp(input_buf(x, y, c) + diff * weight * 255.0f, 0.0f, 255.0f);

    output(x, y, c) = cast<uint8_t>(color);

    output.parallel(y).vectorize(x, 16);
    cv::Mat result(input.size(), input.type());
    Buffer<uint8_t> result_buf(result.data, result.cols, result.rows, 3);
    output.realize(result_buf);

    return result;
}
