#include "ImageUtils.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
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
                    res.at<uchar>(i, j) = Utils::limitValue(static_cast<int>(round(wAvg)));
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
                    res.at<uchar>(i, j) = Utils::isInBounds(I.at<uchar>(i, j), s1.val[0], s2.val[0]) ? MAX_VAL
                                                                                                     : MIN_VAL;
                }
            }
            break;
        case 3:
            cv::Mat_<cv::Vec3b> _I = I;
            for (int i = 0; i < I.rows; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    res.at<uchar>(i, j) = Utils::isInBounds(_I(i, j)[0], s1.val[0], s2.val[0]) &&
                                          Utils::isInBounds(_I(i, j)[1], s1.val[1], s2.val[1]) &&
                                          Utils::isInBounds(_I(i, j)[2], s1.val[2], s2.val[2]) ? MAX_VAL : MIN_VAL;
                }
            }
            break;
    }
    return res;
}


int ImageUtils::floodFill(cv::Mat &I, const cv::Point start, int targetColor, int replacementColor) {
    CV_Assert(I.depth() != sizeof(uchar));

    switch (I.channels()) {
        case 1:
            return floodFillImpl(I, start, targetColor, replacementColor);
    }
    return 0;
}

int ImageUtils::floodFillImpl(cv::Mat &I, const cv::Point &start, int targetColor, int replacementColor) {

    if (targetColor == replacementColor) return -1;
    int startColor = I.at<uchar>(start);
    if (startColor != targetColor) return -1;

    int rows = I.rows;
    int cols = I.cols;
    std::function<bool(int, int)> isInRange = [&rows, &cols](int col, int row) {
        return row > 0 && row < rows - 1 && col > 0 && col < cols - 1;
    };

    int areaCounter = 0;
    std::deque<cv::Point> queue;
    I.at<uchar>(start) = replacementColor;
    queue.push_back(start);

    while (!queue.empty()) {
        areaCounter++;
        cv::Point p = queue.front();
        queue.pop_front();

        if (isInRange(p.x, p.y - 1)) {
            cv::Point west = cv::Point(p.x, p.y - 1);
            if (I.at<uchar>(west) == targetColor) {
                I.at<uchar>(west) = replacementColor;
                queue.push_back(west);
            }
        }
        if (isInRange(p.x, p.y + 1)) {
            cv::Point east = cv::Point(p.x, p.y + 1);
            if (I.at<uchar>(east) == targetColor) {
                I.at<uchar>(east) = replacementColor;
                queue.push_back(east);
            }
        }
        if (isInRange(p.x + 1, p.y)) {
            cv::Point north = cv::Point(p.x + 1, p.y);
            if (I.at<uchar>(north) == targetColor) {
                I.at<uchar>(north) = replacementColor;
                queue.push_back(north);
            }
        }
        if (isInRange(p.x - 1, p.y)) {
            cv::Point south = cv::Point(p.x - 1, p.y);
            if (I.at<uchar>(south) == targetColor) {
                I.at<uchar>(south) = replacementColor;
                queue.push_back(south);
            }
        }
    }

    return areaCounter;
}

cv::Mat ImageUtils::bitwise_xor(const cv::Mat &I1, const cv::Mat &I2) {
    CV_Assert(I1.rows == I2.rows && I1.cols == I2.cols);
    cv::Mat res(I1.rows, I1.cols, CV_8UC1, cv::Scalar(0));
    for (int i = 1; i < I1.rows - 1; ++i) {
        for (int j = 1; j < I2.cols - 1; ++j) {
            if (I1.at<uchar>(i, j) != I2.at<uchar>(i, j)) {
                res.at<uchar>(i, j) = MAX_VAL;
            }
        }
    }
    return res;
}

cv::Mat ImageUtils::bitwise_or(const cv::Mat &I1, const cv::Mat &I2) {
    CV_Assert(I1.rows == I2.rows && I1.cols == I2.cols);
    cv::Mat res(I1.rows, I1.cols, CV_8UC1, cv::Scalar(0));
    for (int i = 1; i < I1.rows - 1; ++i) {
        for (int j = 1; j < I2.cols - 1; ++j) {
            if (I1.at<uchar>(i, j) == MAX_VAL || I2.at<uchar>(i, j) == MAX_VAL) {
                res.at<uchar>(i, j) = MAX_VAL;
            }
        }
    }
    return res;
}

int ImageUtils::calcPerimeter(const cv::Mat &I, int color, int backgroundColor) {
    int perimeter = 0;
    for (int i = 1; i < I.rows - 1; ++i) {
        for (int j = 1; j < I.cols - 1; ++j) {
            if (I.at<uchar>(i, j) == color) {
                if (I.at<uchar>(i - 1, j) == backgroundColor || I.at<uchar>(i + 1, j) == backgroundColor ||
                    I.at<uchar>(i, j - 1) == backgroundColor || I.at<uchar>(i, j + 1) == backgroundColor) {
                    perimeter++;
                }
            }
        }
    }
    return perimeter;
}

double ImageUtils::calcMoment(const cv::Mat &I, int p, int q, int color) {
    double moment = 0;
    for (int i = 0; i < I.rows; ++i) {
        for (int j = 0; j < I.cols; ++j) {
            if (I.at<uchar>(i, j) == color) {
                moment += pow(i, p) * pow(j, q);
            }
        }
    }
    return moment;
}

std::map<std::string, double> ImageUtils::calcMomentums(const cv::Mat &I, int color) {
    double m00 = calcMoment(I, 0, 0, color);
    double m01 = calcMoment(I, 0, 1, color);
    double m10 = calcMoment(I, 1, 0, color);
    double m11 = calcMoment(I, 1, 1, color);
    double m20 = calcMoment(I, 2, 0, color);
    double m02 = calcMoment(I, 0, 2, color);

    double M20 = m20 - (m10 * m10) / m00;
    double M02 = m02 - (m01 * m01) / m00;
    double M11 = m11 - ((m10 * m01) / m00);

    double M1 = (M20 + M02) / (m00 * m00);
    double M7 = (M20 * M02 - M11 * M11) / pow(m00, 4);

    std::map<std::string, double> result;
    result.insert(std::make_pair("M1", M1));
    result.insert(std::make_pair("M7", M7));
    return result;
}

int ImageUtils::calcArea(const cv::Mat &I, int color) {
    int area = 0;
    for (int i = 0; i < I.rows; ++i) {
        for (int j = 0; j < I.cols; ++j) {
            if (I.at<uchar>(i, j) == color) {
                ++area;
            }
        }
    }
    return area;
}

double ImageUtils::calcW3(int area, int perimeter) {
    double pi = 3.14159265;
    return (static_cast<double>(perimeter) / (2.0 * sqrt(pi * static_cast<double>(area)))) - 1.0;
}

std::pair<int, int> ImageUtils::calcWidthHeight(const cv::Mat &I, int color) {
    int firstX = -1;
    int firstY = -1;
    int lastX = -1;
    int lastY = -1;
    for (int i = 0; i < I.rows; ++i) {
        int areaOfRow = ImageUtils::calcArea(I.row(i), color);
        if (firstX == -1 && areaOfRow > 0) {
            firstX = i;
        }
        if (firstX != -1 && areaOfRow > 0) {
            lastX = i;
        }
    }
    for (int i = 0; i < I.cols; ++i) {
        int areaOfCol = ImageUtils::calcArea(I.col(i), color);
        if (firstY == -1 && areaOfCol > 0) {
            firstY = i;
        }
        if (firstY != -1 && areaOfCol > 0) {
            lastY = i;
        }
    }
    int width = lastY - firstY;
    int height = lastX - firstX;

    return std::make_pair(width, height);
}

cv::Rect ImageUtils::boundingRectOfObject(const cv::Mat &I, int color) {
    int firstX = -1;
    int firstY = -1;
    int lastX = -1;
    int lastY = -1;
    for (int i = 0; i < I.rows; ++i) {
        int areaOfRow = ImageUtils::calcArea(I.row(i), color);
        if (firstX == -1 && areaOfRow > 0) {
            firstX = i;
        }
        if (firstX != -1 && areaOfRow > 0) {
            lastX = i;
        }
    }
    for (int i = 0; i < I.cols; ++i) {
        int areaOfCol = ImageUtils::calcArea(I.col(i), color);
        if (firstY == -1 && areaOfCol > 0) {
            firstY = i;
        }
        if (firstY != -1 && areaOfCol > 0) {
            lastY = i;
        }
    }
    int width = lastY - firstY;
    int height = lastX - firstX;
    return cv::Rect(firstY, firstX, width, height);
}

cv::Mat ImageUtils::imageWithMask(const cv::Mat &I, const cv::Rect &mask) {
    cv::Mat result = cv::Mat(I.rows, I.cols, CV_8UC1, cv::Scalar(0));
    for (int i = mask.y; i < mask.y + mask.height; ++i) {
        for (int j = mask.x; j < mask.x + mask.width; ++j) {
            result.at<uchar>(i, j) = I.at<uchar>(i, j);
        }
    }
    return result;
}
