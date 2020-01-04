#include "ImageUtils.h"

#include <algorithm>
#include "Constants.h"

cv::Mat ImageUtils::changeContrast(cv::Mat &I, float percent) {
    CV_Assert(I.depth() != sizeof(uchar));
    switch (I.channels()) {
        case 1:
            for (int i = 0; i < I.rows; ++i)
                for (int j = 0; j < I.cols; ++j)
                    I.at<uchar>(i, j) = limitValue(I.at<uchar>(i, j) * percent);
            break;
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    for (int n = 0; n < 3; ++n) {
                        _I(i, j)[n] = limitValue(_I(i, j)[n] * percent);
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
                    I.at<uchar>(i, j) = limitValue(I.at<uchar>(i, j) + amount);
            break;
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    for (int n = 0; n < 3; ++n) {
                        _I(i, j)[n] = limitValue(_I(i, j)[n] + amount);
                    }
                }
            }
            I = _I;
            break;
    }
    return I;
}

int ImageUtils::limitValue(int val) {
    if (val > MAX_VAL) {
        return MAX_VAL;
    } else if (val < MIN_VAL) {
        return MIN_VAL;
    } else {
        return val;
    }
}

cv::Mat ImageUtils::changeRGBToGray(cv::Mat &I) {
    CV_Assert(I.depth() != sizeof(uchar));
    switch (I.channels()) {
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            float wAvg;
            int newValue = 0;
            for (int i = 0; i < I.rows; ++i)
                for (int j = 0; j < I.cols; ++j) {
                    wAvg = 0.0f;
                    wAvg += (_I(i, j)[0] * GRAY_B + _I(i, j)[1] * GRAY_G + _I(i, j)[2] * GRAY_R);
                    newValue = limitValue(wAvg);
                    for (int n = 0; n <= 2; ++n) {
                        _I(i, j)[n] = newValue;
                    }
                }
            I = _I;
            break;
    }
    return I;
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
