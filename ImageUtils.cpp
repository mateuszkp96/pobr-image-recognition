#include "ImageUtils.h"
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
