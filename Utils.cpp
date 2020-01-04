#include "Utils.h"
#include "Constants.h"

int Utils::limitValue(int val) {
    if (val > MAX_VAL) {
        return MAX_VAL;
    } else if (val < MIN_VAL) {
        return MIN_VAL;
    } else {
        return val;
    }
}

bool Utils::isInInterval(int val, int min, int max) {
    return val >= min && val <= max;
}
