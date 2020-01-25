#ifndef POBR_IMAGEUTILS_H
#define POBR_IMAGEUTILS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>

class ImageUtils {
public:
    static cv::Mat changeContrast(cv::Mat &I, float percent);

    static cv::Mat changeBrightness(cv::Mat &I, int amount);

    static cv::Mat convertRGBToGray(cv::Mat &I);

    static cv::Mat convertRGBToHSV(cv::Mat &I);

    static cv::Mat rankFilter(cv::Mat &I, int kernelSize, int index);

    static cv::Mat inRange(cv::Mat &I, const cv::Scalar &s1, const cv::Scalar &s2);

    static int floodFill(cv::Mat &I, cv::Point start, int targetColor, int replacementColor);

    static cv::Mat bitwise_xor(const cv::Mat &I1, const cv::Mat &I2);

    static cv::Mat bitwise_or(const cv::Mat &I1, const cv::Mat &I2);

    static int calcPerimeter(const cv::Mat &I, int color, int backgroundColor);

    static std::map<std::string, double> calcMomentums(const cv::Mat &I, int color);

    static int calcArea(const cv::Mat &I, int color);

    static double calcW3(int area, int perimeter);

    static double calcMoment(const cv::Mat &I, int p, int q, int color);

    static std::pair<int, int> calcWidthHeight(const cv::Mat &I, int color);

    static cv::Rect boundingRectOfObject(const cv::Mat &I, int color);

    static cv::Mat imageWithMask(const cv::Mat &I, const cv::Rect &mask);

private:

    static int floodFillImpl(cv::Mat &I, const cv::Point &start, int targetColor, int replacementColor);

};

#endif //POBR_IMAGEUTILS_H
