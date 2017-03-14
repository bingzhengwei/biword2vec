#include "src/sampler.h"
#include <algorithm>

BaseSampler::BaseSampler(
    std::vector<std::pair<size_t, double> >& data_weights
) {
}

BaseSampler::~BaseSampler() { }

void BaseSampler::seed(unsigned val) {
    rand_generator_.seed(val);
}


AliasSampler::AliasSampler(
    std::vector<std::pair<size_t, double> >& data_weights
) : BaseSampler(data_weights),
    uniform_int_dist_(0, data_weights.size() - 1),
    uniform_real_dist_(0, std::nextafter(1, std::numeric_limits<double>::max())) {
    Init(data_weights);
}

AliasSampler::~AliasSampler() {
}

bool AliasSampler::Init(
    std::vector<std::pair<size_t, double> >& data_weights
) {
    size_t n = data_weights.size();
    if (n == 0) {
        return false;
    }

    data_index_.resize(n);
    alias_.resize(n);
    alias_prob_.resize(n);

    std::vector<double> probs(n, 0);
    std::vector<size_t> smaller(n, 0);
    std::vector<size_t> larger(n, 0);

    double sum = 0;
    for (size_t i = 0; i < data_weights.size(); ++i) {
        data_index_[i] = data_weights[i].first;
        sum += data_weights[i].second;
    }

    // Normalise given probabilities
    for (size_t i = 0; i < data_weights.size(); ++i) {
        probs[i] = data_weights[i].second * n / sum;
    }

    // Set separate index lists for small and large probabilities:
    int64_t num_smaller = 0;
    int64_t num_larger = 0;
    for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; --i) {
        if (util_less<double>(probs[i], 1)) {
            smaller[num_smaller++] = i;
        } else {
            larger[num_larger++] = i;
        }
    }

    // Work through index lists
    while (num_smaller && num_larger) {
        size_t l = smaller[--num_smaller]; // Schwarz's l
        size_t g = larger[--num_larger]; // Schwarz's g
        alias_prob_[l] = probs[l];
        alias_[l] = g;
        probs[g] = probs[g] + probs[l] - 1;
        if (util_less<double>(probs[g], 1)) {
            smaller[num_smaller++] = g;
        } else {
            larger[num_larger++] = g;
        }
    }

    while (num_larger) {
        alias_prob_[larger[--num_larger]] = 1;
    }

    while (num_smaller) {
        // can only happen through numeric instability
        alias_prob_[smaller[--num_smaller]] = 1;
    }

    return true;
}

size_t AliasSampler::draw() {
    size_t idx = uniform_int_dist_(rand_generator_);
    double rand_prob = uniform_real_dist_(rand_generator_);
    if (util_less<double> (rand_prob, alias_prob_[idx])) {
        return idx;
    } else {
        return alias_[idx];
    }
}

size_t AliasSampler::sampling() {
    size_t idx = draw();
    if (idx >= data_index_.size()) {
        idx = data_index_.size() - 1;
    }

    return data_index_[idx];
}


MultinomialSampler::MultinomialSampler(
    std::vector<std::pair<size_t, double> >& data_weights
) : BaseSampler(data_weights),
    uniform_dist_(0, std::nextafter(1, std::numeric_limits<double>::max())) {

    double total_weight = 0.0;
    std::for_each(
        data_weights.begin(),
        data_weights.end(),
        [&] (const std::pair<size_t, double>& s) {
            total_weight += s.second;
        }
    );

    multinomial_dist_.resize(data_weights.size());
    data_index_.resize(data_weights.size());
    double cur_weight = 0.0;

    for (size_t i = 0; i < data_weights.size(); ++i) {
        cur_weight += data_weights[i].second;
        multinomial_dist_[i] = cur_weight / total_weight;
        data_index_[i] = data_weights[i].first;
    }
}

MultinomialSampler::~MultinomialSampler() { }

double MultinomialSampler::random_impl() {
    return uniform_dist_(rand_generator_);
}

size_t MultinomialSampler::sampling() {
    double rand_prob = random_impl();

    auto it = std::upper_bound(
        multinomial_dist_.begin(),
        multinomial_dist_.end(),
        rand_prob
    );

    size_t pos = it - multinomial_dist_.begin();
    if (pos >= multinomial_dist_.size()) {
        pos = multinomial_dist_.size() - 1;
    }
    return data_index_[pos];
}

RandomSampler::RandomSampler(
    std::vector<std::pair<size_t, double> >& data_weights
) : BaseSampler(data_weights), uniform_dist_(0, data_weights.size() - 1) {
    data_index_.resize(data_weights.size());
    for (size_t i = 0; i < data_weights.size(); ++i) {
        data_index_[i] = data_weights[i].first;
    }
}

RandomSampler::~RandomSampler() {
}

size_t RandomSampler::sampling() {
    size_t idx = uniform_dist_(rand_generator_);
    if (idx >= data_index_.size()) {
        idx = data_index_.size() - 1;
    }

    return data_index_[idx];
}

/* vim: set ts=4 sw=4 tw=0 et :*/
