#include "Processor.h"
#include "ImageUtils.h"
#include "Utils.h"
#include <iostream>
#include <opencv2/imgproc.hpp> // to draw rectangle around logo


Processor::Processor() : blue_min_(cv::Scalar(95, 100, 0)), blue_max_(cv::Scalar(107, 255, 150)),
                         white_min_(cv::Scalar(0, 0, 0)), white_max_(cv::Scalar(180, 50, 120)),
                         black_min_(cv::Scalar(0, 0, 150)), black_max_(cv::Scalar(180, 255, 255)) {

}

void Processor::processImages(const std::vector<std::string> &names) {
    for (const std::string &name : names) {
        std::cout << name << std::endl;
        cv::Mat source = cv::imread(name);
        source = ImageUtils::rankFilter(source, 3, 4);

        cv::Mat sourceClone = source.clone();
        cv::Mat hsvImage = ImageUtils::convertRGBToHSV(sourceClone);
        cv::Mat blueImg = ImageUtils::inRange(hsvImage, blue_min_, blue_max_); //hsv
        cv::Mat whiteImg = ImageUtils::inRange(hsvImage, white_min_, white_max_); //hsv
        cv::Mat blackImg = ImageUtils::inRange(hsvImage, black_min_, black_max_); //hsv

//        cv::imshow("Source", source);
//        cv::imshow("HSV", hsvImage);
//        cv::imshow("blue", blueImg);
//        cv::imshow("black", blackImg);
//        cv::imshow("white", whiteImg);

        auto blue_features = calculateObjectFeatures(blueImg, 255, 0);
        auto quartersBlue = findQuarters(blue_features);

        auto foundLogoRects = processFeatures(blue_features, whiteImg, blackImg);

        for (const auto &rect : foundLogoRects) {
            cv::rectangle(source, rect, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("Output", source);
        cv::waitKey(-1);
    }
}

std::vector<ObjectFeatures> Processor::calculateObjectFeatures(cv::Mat &I, int color, int backgroundColor) {
    std::vector<ObjectFeatures> result;
    cv::Mat input = I.clone();
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            if (input.at<uchar>(i, j) == color) {
                cv::Mat before = input.clone();
                int area = ImageUtils::floodFill(input, cv::Point(j, i), color, backgroundColor);
                if (area > 20) {
                    cv::Mat object = ImageUtils::bitwise_xor(before, input);
                    result.push_back(ObjectFeatures(object, color, backgroundColor));
                }
            }
        }
    }
    return result;
}

std::vector<cv::Rect>
Processor::processFeatures(const std::vector<ObjectFeatures> &input, cv::Mat &white, cv::Mat &black) {
    CV_Assert(white.rows == black.rows && white.cols == black.cols);

    std::vector<cv::Rect> result;

    auto filterFunc = [&](const ObjectFeatures &f) {
        int firstX = Utils::boundValue(f.x_center - 2 * f.width, 0, white.rows - 1);
        int lastX = Utils::boundValue(f.x_center + 2 * f.width, 0, white.rows - 1);
        int firstY = Utils::boundValue(f.y_center - 2 * f.height, 0, white.cols - 1);
        int lastY = Utils::boundValue(f.y_center + 2 * f.height, 0, white.cols - 1);
        int width = lastY - firstY;
        int height = lastX - firstX;
        cv::Rect boundingRect = cv::Rect(firstY, firstX, width, height);
        cv::Mat cutWhite = white(boundingRect);
        cv::Mat cutBlack = black(boundingRect);

        int areaW = ImageUtils::calcArea(cutWhite, 255);
        int areaB = ImageUtils::calcArea(cutBlack, 255);

        bool featurePredicate = f.aspect <= 1.6 && f.aspect >= 0.4;
        bool areaPredicate = f.area < areaB && f.area < areaW && f.area > 5;
        int otherFeaturesInsideRect = std::count_if(input.begin(), input.end(), [&](const ObjectFeatures &f) {
            return boundingRect.contains(cv::Point(f.y_center, f.x_center));
        });
        return featurePredicate && areaPredicate && otherFeaturesInsideRect > 0;
    };

    std::vector<ObjectFeatures> filtered;
    std::copy_if(input.begin(), input.end(), std::back_inserter(filtered), filterFunc);

    std::map<int, ObjectFeatures> blueObjects;
    std::transform(filtered.begin(), filtered.end(), std::inserter(blueObjects, blueObjects.begin()),
                   [](const ObjectFeatures &f) { return std::make_pair(f.id, f); });

    auto closestObjectFunc = [&](const ObjectFeatures &f) {
        auto best = std::make_pair(-1, std::numeric_limits<double>::max());
        std::for_each(blueObjects.begin(), blueObjects.end(), [&](const std::pair<int, ObjectFeatures> &pair) {
            if (pair.first != f.id) {
                double distance = Utils::distance(f.x_center, f.y_center, pair.second.x_center, pair.second.y_center);
                if (distance < best.second &&
                    Utils::isInBounds(f.width / static_cast<double>(pair.second.width), 0.6, 1.4)) {
                    best.first = pair.first;
                    best.second = distance;
                }
            }
        });
        return best;
    };

    std::map<int, int> blue_pairs;
    std::for_each(blueObjects.begin(), blueObjects.end(), [&](const std::pair<int, ObjectFeatures> &pair) {
        auto closest = closestObjectFunc(pair.second);
        if (closest.first != -1) {
            blue_pairs.insert(std::make_pair(pair.first, closest.first));
        }
    });

    auto pairsConnected = getPairsConnected(blue_pairs);
    std::cout << "Number of pairs: " << pairsConnected.size() << std::endl;

//    std::for_each(blueObjects.begin(), blueObjects.end(), [&](const std::pair<int, ObjectFeatures> &pair) {
//        cv::imshow("Blue object", pair.second.object);
//        std::cout << "id: " << pair.second.id << std::endl;
//        cv::waitKey(-1);
//    });

    for (auto pair : pairsConnected) {
        auto firstObj = blueObjects.find(pair.first)->second;
        auto secondObj = blueObjects.find(pair.second)->second;

        cv::Mat sum = ImageUtils::bitwise_or(firstObj.object, secondObj.object);
        cv::Rect boundingRect = ImageUtils::boundingRectOfObject(sum, 255);

        cv::Mat cutWhite = ImageUtils::imageWithMask(white, boundingRect);
        ObjectFeatures featWhite = ObjectFeatures(cutWhite, 255, 0);

        double percent = featWhite.area / static_cast<double>(boundingRect.area());
        if (percent > 0.15 && percent < 0.55) {
            std::cout << "White percent correct" << std::endl;
            int new_width = 1.6 * boundingRect.width;
            int new_height = 1.6 * boundingRect.height;

            int new_x = Utils::boundValue(boundingRect.x - new_width * 0.3 / 2.0, 0, white.rows);
            int new_y = Utils::boundValue(boundingRect.y - new_height * 0.3 / 2.0, 0, white.cols);

            cv::Rect rectForBlack(new_x, new_y, new_width, new_height);
            cv::Mat cutBlack = ImageUtils::imageWithMask(black, rectForBlack);
            ObjectFeatures blackObject = ObjectFeatures(cutBlack, 255, 0);

            if (blackObject.W3 > 4) {
                continue;
            }

            if (rectForBlack.contains(blackObject.getCenter()) &&
                Utils::isInBounds(rectForBlack.width / static_cast<double>(rectForBlack.height), 0.8, 1.2)) {
                result.push_back(rectForBlack);
                std::cout << "LOGO FOUND!!!" << std::endl;
            }
        }
    }

    return result;
}

std::vector<ObjectFeatures> Processor::findQuarters(const std::vector<ObjectFeatures> &input) {
    auto isQuarterCandidate = [](const ObjectFeatures &f) {
        return f.area > 20 && f.aspect > 0.5 && f.aspect < 2;
    };
    std::vector<ObjectFeatures> filtered;
    std::copy_if(input.begin(), input.end(), std::back_inserter(filtered), isQuarterCandidate);
    return filtered;
}

std::vector<std::pair<int, int>> Processor::getPairsConnected(const std::map<int, int> &pairsMap) {
    std::map<int, bool> paired;
    std::vector<std::pair<int, int>> pairs;

    std::for_each(pairsMap.begin(), pairsMap.end(), [&](std::pair<int, int> pair) {
        paired.insert(std::make_pair(pair.first, false));
    });

    std::for_each(pairsMap.begin(), pairsMap.end(), [&](std::pair<int, int> pair) {

        if (!paired.at(pair.first)) {
            auto secondElem = pairsMap.find(pair.second);
            if (secondElem != pairsMap.end()) {
                if (secondElem->second == pair.first) {
                    // is pair
                    paired[pair.first] = true;
                    paired[secondElem->first] = true;
                    pairs.push_back(std::make_pair(pair.first, pair.second));
                }
            }
        }
    });

    return pairs;
}


