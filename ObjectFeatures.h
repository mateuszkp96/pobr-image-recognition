#ifndef POBR_OBJECTFEATURES_H
#define POBR_OBJECTFEATURES_H

#include <opencv2/core/core.hpp>

class ObjectFeatures {
public:
    const int id;
    int perimeter;
    int area;
    int x_center, y_center;
    int width, height;
    double aspect;
    double W3;
    double M1, M2, M3, M4, M5, M6, M7;
    cv::Mat object;

    ObjectFeatures(const cv::Mat &I, int color, int backgroundColor);

    void print();

    cv::Point getCenter();

private:
    static int id_counter;
};


#endif //POBR_OBJECTFEATURES_H
