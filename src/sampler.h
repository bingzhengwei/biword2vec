#ifndef  SRC_SAMPLER_H
#define  SRC_SAMPLER_H

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "src/util.h"

class BaseSampler {
public:
    BaseSampler(
        std::vector<std::pair<size_t, double> >& data_weights
    );

    virtual ~BaseSampler();

public:
    virtual void seed(unsigned val);
    virtual size_t sampling() = 0;

protected:
    std::default_random_engine rand_generator_;
};

class AliasSampler : public BaseSampler {
public:
    AliasSampler(
        std::vector<std::pair<size_t, double> >& data_weights
    );

    virtual ~AliasSampler();

public:
    size_t sampling();

protected:
    bool Init(
        std::vector<std::pair<size_t, double> >& data_weights
    );

    size_t draw();

private:
    std::vector<size_t> alias_;
    std::vector<double> alias_prob_;
    std::vector<size_t> data_index_;

    std::uniform_int_distribution<size_t> uniform_int_dist_;
    std::uniform_real_distribution<double> uniform_real_dist_;
};

class MultinomialSampler : public BaseSampler {
public:
    MultinomialSampler(
        std::vector<std::pair<size_t, double> >& data_weights
    );

    virtual ~MultinomialSampler();

public:
    size_t sampling();

protected:
    double random_impl();

protected:
    std::uniform_real_distribution<double> uniform_dist_;

private:
    std::vector<double> multinomial_dist_;
    std::vector<size_t> data_index_;
};

class RandomSampler : public BaseSampler {
public:
    RandomSampler(
        std::vector<std::pair<size_t, double> >& data_weights
    );

    virtual ~RandomSampler();

public:
    virtual size_t sampling();

protected:
    std::uniform_int_distribution<size_t> uniform_dist_;
    std::vector<size_t> data_index_;
};

#endif // SRC_SAMPLER_H
/* vim: set ts=4 sw=4 tw=0 et :*/
