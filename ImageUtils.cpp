#include "ImageUtils.h"

#include <algorithm>
#include <cmath>
#include "Utils.h"
#include "Constants.h"

cv::Mat ImageUtils::changeContrast(cv::Mat &I, float percent) {
    CV_Assert(I.depth() != sizeof(uchar));
    switch (I.channels()) {
        case 1:
            for (int i = 0; i < I.rows; ++i)
                for (int j = 0; j < I.cols; ++j)
                    I.at<uchar>(i, j) = Utils::limitValue(I.at<uchar>(i, j) * percent);
            break;
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    for (int n = 0; n < 3; ++n) {
                        _I(i, j)[n] = Utils::limitValue(_I(i, j)[n] * percent);
                    }
                }
            }
            I = _I;
            break;
    }
    return I;
}

cv::Mat ImageUtils::changeBrightness(cv::Mat &I, int amount) {
    CV_Assert(I.depth() != sizeof(uchar));
    switch (I.channels()) {
        case 1:
            for (int i = 0; i < I.rows; ++i)
                for (int j = 0; j < I.cols; ++j)
                    I.at<uchar>(i, j) = Utils::limitValue(I.at<uchar>(i, j) + amount);
            break;
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    for (int n = 0; n < 3; ++n) {
                        _I(i, j)[n] = Utils::limitValue(_I(i, j)[n] + amount);
                    }
                }
            }
            I = _I;
            break;
    }
    return I;
}

cv::Mat ImageUtils::convertRGBToGray(cv::Mat &I) {
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat res(I.rows, I.cols, CV_8UC1);
    switch (I.channels()) {
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            double wAvg;
            for (int i = 0; i < I.rows; ++i)
                for (int j = 0; j < I.cols; ++j) {
                    wAvg = (_I(i, j)[0] * GRAY_B + _I(i, j)[1] * GRAY_G + _I(i, j)[2] * GRAY_R);
                    res.at<uchar>(i,j) = Utils::limitValue(static_cast<int>(round(wAvg)));
                }
            break;
    }
    return res;
}

cv::Mat ImageUtils::rankFilter(cv::Mat &I, const int kernelSize, const int index) {
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat res(I.rows, I.cols, CV_8UC3);
    switch (I.channels()) {
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            cv::Mat_<cv::Vec3b> _R = res;
            int offset = kernelSize / 2;
            std::vector<std::pair<int, int>> values(kernelSize * kernelSize);

            int x, y, count, winnerIndex;
            for (int i = offset; i < I.rows - offset; ++i) {
                for (int j = offset; j < I.cols - offset; ++j) {
                    count = 0;
                    for (int m = -offset; m < -offset + kernelSize; ++m) {
                        for (int n = -offset; n < -offset + kernelSize; ++n) {
                            x = i + m;
                            y = j + n;
                            values[count].first = x * I.cols + y;
                            values[count].second = (_I(x, y)[0] + _I(x, y)[1] + _I(x, y)[2]) / 3;
                            count++;
                        }
                    }
                    std::nth_element(values.begin(), values.begin() + index, values.end(),
                                     [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                                         return a.second < b.second;
                                     });
                    winnerIndex = values[index].first;
                    cv::Vec3b winner = _I(winnerIndex / I.cols, winnerIndex % I.cols);
                    _R(i, j)[0] = winner[0];
                    _R(i, j)[1] = winner[1];
                    _R(i, j)[2] = winner[2];
                }
            }
            res = _R;
            break;
    }
    return res;
}

cv::Mat ImageUtils::inRange(cv::Mat &I, const cv::Scalar &s1, const cv::Scalar &s2) {
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat res(I.rows, I.cols, CV_8UC1);
    switch (I.channels()) {
        case 1:
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    res.at<uchar>(i, j) = Utils::isInInterval(I.at<uchar>(i, j), s1.val[0], s2.val[0]) ? MAX_VAL : MIN_VAL;
                }
            }
            break;
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    res.at<uchar>(i, j) = Utils::isInInterval(_I(i, j)[0], s1.val[0], s2.val[0]) &&
                                          Utils::isInInterval(_I(i, j)[1], s1.val[1], s2.val[1]) &&
                                          Utils::isInInterval(_I(i, j)[2], s1.val[2], s2.val[2]) ? MAX_VAL : MIN_VAL;
                }
            }
            break;
    }
    return res;
}

cv::Mat ImageUtils::convertRGBToHSV(cv::Mat &I) {
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat res(I.rows, I.cols, CV_8UC3); // H S V
    switch (I.channels()) {
        case 3:
            int minIdx, maxIdx;
            float minVal, maxVal;
            int r, g, b;
            float h, s, v;
            cv::Mat_<cv::Vec3b> _R = res;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    cv::Vec3b intensity = I.at<cv::Vec3b>(i, j);
                    b = intensity[BLUE_IDX];
                    g = intensity[GREEN_IDX],
                    r = intensity[RED_IDX];
                    // max
                    maxIdx = (b < g) ? GREEN_IDX : BLUE_IDX;
                    maxIdx = (intensity[maxIdx] < r) ? RED_IDX : maxIdx;
                    maxVal = intensity[maxIdx];
                    // min
                    minIdx = (b < g) ? BLUE_IDX : GREEN_IDX;
                    minIdx = (intensity[minIdx] < r) ? minIdx : RED_IDX;
                    minVal = intensity[minIdx];

                    v = maxVal;
                    s = v != 0 ? (v - minVal) / v : 0;
                    switch (maxIdx) {
                        case BLUE_IDX:
                            h = 240 + 60 * (r - g) / (v - minVal);
                            break;
                        case GREEN_IDX:
                            h = 120 + 60 * (b - r) / (v - minVal);
                            break;
                        case RED_IDX:
                            h = 60 * (g - b) / (v - minVal);
                            break;
                        default:
                            h = 0;
                            break;
                    }

                    _R(i, j)[HUE_IDX] = h / 2;
                    _R(i, j)[SAT_IDX] = 255 * s;
                    _R(i, j)[VAL_IDX] = 255 * v;
                }
            }
            res = _R;
            break;
    }
    return res;
}
