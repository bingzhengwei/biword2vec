#include "src/util.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>

SigmoidTable::SigmoidTable(size_t table_size) : table_size_(table_size) {
    sigmoid_table_ = new double[table_size_ + 1];
    log_sigmoid_table_ = new double[table_size_ + 1];
    for (size_t i = 0; i < table_size_; ++i) {
        double exp_value =
            static_cast<double>(2 * MAX_EXP_NUM * i) / table_size_ - MAX_EXP_NUM;

        sigmoid_table_[i] = sigmoid<double>(exp_value);
        log_sigmoid_table_[i] = safe_log<double>(sigmoid_table_[i]);
    }
}

SigmoidTable::~SigmoidTable() {
    if (sigmoid_table_) {
        delete [] sigmoid_table_;
    }

    if (log_sigmoid_table_) {
        delete [] log_sigmoid_table_;
    }
}

double SigmoidTable::LogSigmoid(double x) {
    if (util_greater<double>(x, MAX_EXP_NUM)) {
        return 1.;
    } else if (util_less<double>(x, -MAX_EXP_NUM)) {
        return 0.;
    }

    size_t idx = (x + MAX_EXP_NUM) * table_size_ / (MAX_EXP_NUM * 2);
    if (idx >= table_size_) {
        idx = table_size_ - 1;
    }

    return log_sigmoid_table_[idx];
}

double SigmoidTable::operator[] (double x) {
    if (util_greater<double>(x, MAX_EXP_NUM)) {
        return 1.;
    } else if (util_less<double>(x, -MAX_EXP_NUM)) {
        return 0.;
    }

    size_t idx = (x + MAX_EXP_NUM) * table_size_ / (MAX_EXP_NUM * 2);
    if (idx >= table_size_) {
        idx = table_size_ - 1;
    }

    return sigmoid_table_[idx];
}

size_t count_file_lines(const char* path) {
    size_t number_of_lines = 0;

    int n;
    size_t buf_size = 16 * 1024;
    char* buf = new char[buf_size];
    int fd = open(path, O_RDONLY);

    if (fd == -1) {
        delete [] buf;
        return 0;
    }

    while ((n = read(fd, buf, buf_size)) > 0) {
        for (
                char *p = buf;
                (p = reinterpret_cast<char*>(memchr(p, '\n', (buf + n) - p)));
                ++p
        ) {
            ++number_of_lines;
        }
    }

    close(fd);

    delete [] buf;
    return number_of_lines;
}

/* vim: set ts=4 sw=4 tw=0 et :*/
