#include "Utils.h"
#include "Constants.h"
#include <cmath>

int Utils::limitValue(int val) {
    if (val > MAX_VAL) {
        return MAX_VAL;
    } else if (val < MIN_VAL) {
        return MIN_VAL;
    } else {
        return val;
    }
}

bool Utils::isInBounds(double val, double min, double max) {
    return val >= min && val <= max;
}

int Utils::boundValue(int val, int min, int max) {
    if (val > max) return max;
    if (val < min) return min;
    return val;
}

double Utils::distance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}
