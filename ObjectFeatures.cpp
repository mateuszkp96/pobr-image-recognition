#include "ObjectFeatures.h"
#include "ImageUtils.h"

#include <iostream>

void ObjectFeatures::print() {
    std::cout << "S: " << area << '\t' << "L: " << perimeter << '\t' << "W3: " << W3 << '\t'
              << "M1: " << M1 << '\t' << "M7: " << M7 << '\t' << std::endl;
}

ObjectFeatures::ObjectFeatures(const cv::Mat &I, int color, int backgroundColor): id(id_counter++) {
    object = I.clone();
    area = ImageUtils::calcArea(I, color);
    perimeter = ImageUtils::calcPerimeter(I, color, backgroundColor);
    W3 = ImageUtils::calcW3(area, perimeter);

    auto widthHeight = ImageUtils::calcWidthHeight(I, color);
    width = widthHeight.first;
    height = widthHeight.second;
    aspect = width / static_cast<double>(height);

    auto momentums = ImageUtils::calcMomentums(I, color);
    M1 = momentums.at("M1");
    M2 = 0;
    M3 = 0;
    M4 = 0;
    M5 = 0;
    M6 = 0;
    M7 = momentums.at("M7");

    double m00 = ImageUtils::calcMoment(I, 0, 0, color);
    double m01 = ImageUtils::calcMoment(I, 0, 1, color);
    double m10 = ImageUtils::calcMoment(I, 1, 0, color);

    x_center = m10 / m00;
    y_center = m01 / m00;
}

int ObjectFeatures::id_counter = 1;

cv::Point ObjectFeatures::getCenter() {
    return cv::Point(y_center, x_center);
}
