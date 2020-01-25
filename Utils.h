#ifndef POBR_UTILS_H
#define POBR_UTILS_H


class Utils {
public:
    static int limitValue(int val);

    static bool isInBounds(double val, double min, double max);

    static int boundValue(int val, int min, int max);

    static double distance(int x1, int y1, int x2, int y2);
};


#endif //POBR_UTILS_H
