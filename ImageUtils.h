#ifndef POBR_IMAGEUTILS_H
#define POBR_IMAGEUTILS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageUtils {
public:
    static cv::Mat changeContrast(cv::Mat &I, float percent);

    static cv::Mat changeBrightness(cv::Mat &I, int amount);

    static cv::Mat convertRGBToGray(cv::Mat &I);

    static cv::Mat rankFilter(cv::Mat &I, int kernelSize, int index);

    static cv::Mat inRange(cv::Mat &I, const cv::Scalar &s1, const cv::Scalar &s2);

};


#endif //POBR_IMAGEUTILS_H
