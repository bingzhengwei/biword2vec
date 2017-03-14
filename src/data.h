#ifndef SRC_DATA_H
#define SRC_DATA_H

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/lock.h"
#include "src/sampler.h"
#include "src/util.h"
#include "src/word_table.h"

static const size_t BUF_SIZE = 102400;

template <typename IdType, typename T>
class Sample {
public:
    Sample();
    Sample(IdType s, IdType t, T w = 1.);
    ~Sample();

    IdType source() { return source_; }
    IdType target() { return target_; }
    T weight() { return weight_; }

    IdType source() const { return source_; }
    IdType target() const { return target_; }
    T weight() const { return weight_; }

    void set_source(IdType id) { source_ = id; }
    void set_target(IdType id) { target_ = id; }
    void set_weight(T weight) { weight_ = weight; }

private:
    IdType source_;
    IdType target_;
    T weight_;
};

template <typename IdType, typename T>
Sample<IdType, T>::Sample() :
        source_(0), target_(0), weight_(0) {
}

template <typename IdType, typename T>
Sample<IdType, T>::Sample(IdType s, IdType t, T w) :
        source_(s), target_(t), weight_(w) {
}

template <typename IdType, typename T>
Sample<IdType, T>::~Sample() {
}

template <typename IdType, typename T>
class DataManager {
public:
    DataManager();
    virtual ~DataManager();

    bool load_data(const std::string& path, size_t num_threads = 1);

    const Sample<IdType, T>* SampleAt(size_t pos);

    std::string SourceWord(IdType pos);

    std::string TargetWord(IdType pos);

    BaseSampler* build_data_sampler(unsigned seed = 1);

    BaseSampler* build_target_sampler(unsigned seed = 1, double weight_exp = 0);

    inline size_t size() {
        return samples_.size();
    }

    inline size_t source_size() {
        return source_words_.size();
    }

    inline size_t target_size() {
        return target_words_.size();
    }

private:
    bool parse_data(char* input_buf, Sample<IdType, T>* sample);

private:
    std::vector<Sample<IdType, T> > samples_;
    WordTable source_words_;
    WordTable target_words_;
};

template <typename IdType, typename T>
DataManager<IdType, T>::DataManager() {
}

template <typename IdType, typename T>
DataManager<IdType, T>::~DataManager() {
}

template <typename IdType, typename T>
bool DataManager<IdType, T>::load_data(
        const std::string& path,
        size_t num_threads
    ) {
    FILE* file_desc = fopen(path.c_str(), "r");

    if (!file_desc) {
        return false;
    }

    size_t num_of_lines = count_file_lines(path.c_str());
    samples_.reserve(num_of_lines);
    source_words_.reserve(num_of_lines);
    target_words_.reserve(num_of_lines);

    SpinLock file_lock;
    SpinLock sample_lock;
    auto parser_thread = [&] (size_t i) {
        char* thread_buf = new char[BUF_SIZE];
        char* ptr = nullptr;

        while (1) {
            {
                std::lock_guard<SpinLock> lock(file_lock);
                ptr = fgets(thread_buf, BUF_SIZE - 1, file_desc);
            }

            if (ptr == nullptr) {
                break;
            }

            thread_buf[BUF_SIZE - 1] = 0;
            Sample<IdType, T> sample;
            bool ret = parse_data(thread_buf, &sample);

            if (ret) {
                std::lock_guard<SpinLock> lock(sample_lock);
                samples_.push_back(std::move(sample));
            }
        }

        delete [] thread_buf;
    };

    util_parallel_run(parser_thread, num_threads);
    fclose(file_desc);
    return true;
}

template <typename IdType, typename T>
const Sample<IdType, T>* DataManager<IdType, T>::SampleAt(size_t pos) {
    if (pos >= samples_.size()) {
        return nullptr;
    }

    return &samples_[pos];
}

template <typename IdType, typename T>
BaseSampler* DataManager<IdType, T>::build_data_sampler(unsigned seed) {
    if (samples_.size() == 0) {
        return nullptr;
    }

    std::vector<std::pair<size_t, double> > data_weights;
    for (size_t i = 0; i < samples_.size(); ++i) {
        data_weights.push_back(
            std::pair<size_t, double> (
                i,
                samples_[i].weight()
            )
        );
    }

    AliasSampler* sampler = new AliasSampler(data_weights);
    sampler->seed(seed);
    return sampler;
}

template <typename IdType, typename T>
BaseSampler* DataManager<IdType, T>::build_target_sampler(unsigned seed, double weight_exp) {
    if (samples_.size() == 0) {
        return nullptr;
    }

    std::unordered_map<IdType, double> target_weights;
    std::vector<std::pair<size_t, double> > data_weights;
    for (size_t i = 0; i < samples_.size(); ++i) {
        target_weights[samples_[i].target()] += samples_[i].weight();
    }

    for (
        auto iter = target_weights.begin();
        iter != target_weights.end();
        ++iter
    ) {
        double weight = pow(iter->second, weight_exp);
        data_weights.push_back(
            std::pair<size_t, double>(iter->first, weight)
        );
    }

    if (util_equal<double> (weight_exp, 0)) {
        RandomSampler* sampler = new RandomSampler(data_weights);
        sampler->seed(seed);
        return sampler;
    } else {
        AliasSampler* sampler = new AliasSampler(data_weights);
        sampler->seed(seed);
        return sampler;
    }
}

template <typename IdType, typename T>
std::string DataManager<IdType, T>::SourceWord(IdType pos) {
    return source_words_.WordAt(pos);
}

template <typename IdType, typename T>
std::string DataManager<IdType, T>::TargetWord(IdType pos) {
    return target_words_.WordAt(pos);
}

template <typename IdType, typename T>
bool DataManager<IdType, T>::parse_data(char* input_buf, Sample<IdType, T>* sample) {
    if (!input_buf || !sample) {
        return false;
    }

    char *ptr;
    char *p1 = strtok_r(input_buf, " \t\r\n", &ptr);
    char *p2 = strtok_r(nullptr, " \t\r\n", &ptr);
    char *p3 = strtok_r(nullptr, " \t\r\n", &ptr);

    if (!p1 || !p2 || !p3) {
        return false;
    }

    T weight = atof(p3);
    if (util_less<T>(weight, 0)) {
        weight = 0.;
    }

    IdType source = source_words_.SearchWord(p1);
    IdType target = target_words_.SearchWord(p2);

    sample->set_weight(weight);
    sample->set_source(source);
    sample->set_target(target);

    return true;
}

#endif // SRC_DATA_H
/* vim: set ts=4 sw=4 tw=0 et :*/
