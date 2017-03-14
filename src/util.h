#ifndef SRC_UTIL_H
#define SRC_UTIL_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <future>
#include <limits>
#include <vector>

const double MAX_EXP_NUM = 20.0;
const size_t DEF_EXP_TABLE_SIZE = 1000;

size_t count_file_lines(const char* path);


class SigmoidTable {
public:
    explicit SigmoidTable(size_t table_size = DEF_EXP_TABLE_SIZE);
    virtual ~SigmoidTable();

    double operator[] (double x);

    double LogSigmoid(double x);

private:
    size_t table_size_;
    double* sigmoid_table_;
    double* log_sigmoid_table_;
};


template <class Func>
void util_parallel_run(const Func& func, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    std::thread *threads = new std::thread[num_threads];
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(func, i);
    }

    for (size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    delete [] threads;
}

template <typename T>
inline bool util_equal(const T v1, const T v2) {
    return std::fabs(v1 - v2) < std::numeric_limits<T>::epsilon();
}

template <typename T>
inline bool util_greater(const T v1, const T v2) {
    if (util_equal(v1, v2)) {
        return false;
    }

    return v1 > v2;
}

template <typename T>
inline int util_cmp(const T v1, const T v2) {
    if (util_equal(v1, v2)) {
        return 0;
    } else if (v1 > v2) {
        return 1;
    } else {
        return -1;
    }
}

template <typename T>
inline bool util_greater_equal(const T v1, const T v2) {
    if (util_equal(v1, v2)) {
        return true;
    }

    return v1 > v2;
}

template <typename T>
inline bool util_less(const T v1, const T v2) {
    if (util_equal(v1, v2)) {
        return false;
    }

    return v1 < v2;
}

template <typename T>
inline bool util_less_equal(const T v1, const T v2) {
    if (util_equal(v1, v2)) {
        return true;
    }

    return v1 < v2;
}

template <typename T>
inline T safe_exp(T x) {
    T max_exp = static_cast<T>(MAX_EXP_NUM);
    return std::exp(std::max(std::min(x, max_exp), -max_exp));
}

template <typename T>
inline T sigmoid(T x) {
    T one = 1.;
    return one / (one + safe_exp(-x));
}

template <typename T>
inline T safe_log(T x) {
    T p = std::max(x, static_cast<T>(1e-10));
    return log(p);
}

#endif // SRC_UTIL_H
/* vim: set ts=4 sw=4 tw=0 et :*/
