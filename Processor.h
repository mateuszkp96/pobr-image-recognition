#ifndef POBR_PROCESSOR_H
#define POBR_PROCESSOR_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <map>
#include <string>
#include "ObjectFeatures.h"

class Processor {
public:
    Processor();

    void processImages(const std::vector<std::string> &names);

private:
    cv::Scalar blue_min_;
    cv::Scalar blue_max_;

    cv::Scalar white_min_;
    cv::Scalar white_max_;

    cv::Scalar black_min_;
    cv::Scalar black_max_;

    std::vector<ObjectFeatures> calculateObjectFeatures(cv::Mat &I, int color, int backgroundColor);

    std::vector<cv::Rect> processFeatures(const std::vector<ObjectFeatures> &input, cv::Mat &white, cv::Mat &black);

    std::vector<ObjectFeatures> findQuarters(const std::vector<ObjectFeatures> &input);

    std::vector<std::pair<int, int>> getPairsConnected(const std::map<int, int> &pairsMap);

};

#endif //POBR_PROCESSOR_H
