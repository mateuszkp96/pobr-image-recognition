#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <map>
#include "Processor.h"

int main() {
    cv::Mat grey;

    std::vector<std::string> names = {"2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",
                                      "11.jpg"};
    std::string prefix = "../images/";
    std::transform(names.begin(), names.end(), names.begin(), [&prefix](const std::string& name) { return prefix + name; });

    Processor processor;
    processor.processImages(names);

    cv::destroyAllWindows();
    return 0;
}
