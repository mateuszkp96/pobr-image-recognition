#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    cv::Mat image = cv::imread("../images/Lena.png");
    cv::imshow("Test", image);
    cv::waitKey(-1);
    return 0;
}
