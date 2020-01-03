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
