#ifndef SRC_BIWORD2VEC_H
#define SRC_BIWORD2VEC_H

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "src/data.h"
#include "src/sampler.h"
#include "src/util.h"

typedef std::function<std::string(size_t id)> name_func_t;

template <typename IdType, typename T>
struct NoiseProbFunctionType {
    typedef std::function<T(IdType)> Type;
};

enum LossType { LOSS_LINE = 0, LOSS_NCE = 1 };

template <typename T>
class BiWord2VecModel {
public:
    BiWord2VecModel(size_t source, size_t target, size_t hidden, T alpha);

    virtual ~BiWord2VecModel();

    bool InitModel(unsigned seed = 1);

    T Predict(size_t source_id, size_t target_id);

    T PredictRaw(size_t source_id, size_t target_id);

    T Update(
        size_t source_id,
        size_t target_id,
        std::vector<size_t>& negative_targets,
        typename NoiseProbFunctionType<size_t, T>::Type noise_prob_func = nullptr,
        T decay = 1.,
        T* buffer = nullptr
    );

    void Save(
        const char* model_path,
        name_func_t source_name = nullptr,
        name_func_t target_name = nullptr
    );

public:
    size_t hidden_size() { return hidden_size_; }
    size_t source_size() { return source_size_; }
    size_t target_size() { return target_size_; }

public:
    T alpha_;
    size_t hidden_size_;

    size_t source_size_;
    size_t target_size_;

    T* source_hidden_;
    T* target_hidden_;

    SigmoidTable sigmoid_table_;
};

template <typename IdType, typename T>
class BiWord2VecTrainer {
public:
    struct TrainingContext {
        BiWord2VecModel<T>* model;
        size_t training_words;
        size_t training_words_actual;
        size_t negative;
        size_t num_threads;
        size_t iteration;
        double logloss;
        size_t logloss_count;
        const std::unordered_map<IdType, T>* target_noise_prob;

        TrainingContext() {
            model = nullptr;
            training_words = 0;
            training_words_actual = 0;
            negative = 0;
            num_threads = 0;
            iteration = 1;
            logloss = 0;
            logloss_count = 0;
            target_noise_prob = nullptr;
        }
    };

public:
    BiWord2VecTrainer();

    virtual ~BiWord2VecTrainer();

    bool Train(
        const char* input_path,
        const char* model_path,
        T alpha,
        size_t hidden_size,
        size_t iteration,
        size_t negative,
        size_t training_words,
        size_t num_threads,
        double weight_neg_sampling,
        WeightType weight_type = WEIGHT_FREQ,
        LossType method = LOSS_LINE,
        unsigned seed = 1
    );

    void TrainThread(
        size_t thread_id,
        TrainingContext* context
    );

public:
    DataManager<IdType, T>* data_manager_;

    BaseSampler* data_sampler_;
    BaseSampler* target_sampler_;
};

template <typename T>
BiWord2VecModel<T>::BiWord2VecModel(
    size_t source,
    size_t target,
    size_t hidden,
    T alpha
) : alpha_(alpha),
    hidden_size_(hidden),
    source_size_(source),
    target_size_(target),
    sigmoid_table_() {
    source_hidden_ = new T[source_size_ * hidden_size_ + 1];
    target_hidden_ = new T[target_size_ * hidden_size_ + 1];
}

template <typename T>
BiWord2VecModel<T>::~BiWord2VecModel() {
    if (source_hidden_) {
        delete [] source_hidden_;
    }

    if (target_hidden_) {
        delete [] target_hidden_;
    }
}

template <typename T>
bool BiWord2VecModel<T>::InitModel(unsigned seed) {
    std::default_random_engine rand_generator;
    std::uniform_real_distribution<T> uniform_dist(-0.5, 0.5);
    rand_generator.seed(seed);

    for (size_t i = 0; i < source_size_ * hidden_size_; ++i) {
        T r = uniform_dist(rand_generator);
        source_hidden_[i] = r / hidden_size_;
    }

    for (size_t i = 0; i < target_size_ * hidden_size_; ++i) {
        T r = uniform_dist(rand_generator);
        target_hidden_[i] = r / hidden_size_;
    }

    return true;
}

template <typename T>
void BiWord2VecModel<T>::Save(const char* path, name_func_t source_name, name_func_t target_name) {
    std::string source_path = std::string(path) + std::string(".source");
    std::string target_path = std::string(path) + std::string(".target");

    FILE* fp = fopen(source_path.c_str(), "w");
    fprintf(fp, "%lu %lu\n", source_size_, hidden_size_);
    for (size_t i = 0; i < source_size_; ++i) {
        if (source_name != nullptr) {
            std::string w = source_name(i);
            fprintf(fp, "%s\t", w.c_str());
        } else {
            fprintf(fp, "%lu\t", i);
        }
        for (size_t j = 0; j < hidden_size_; ++j) {
            const char* tail = (j == hidden_size_ - 1) ? "\n" : " ";
            fprintf(fp, "%lf%s", source_hidden_[i * hidden_size_ + j], tail);
        }
    }
    fclose(fp);

    fp = fopen(target_path.c_str(), "w");
    fprintf(fp, "%lu %lu\n", target_size_, hidden_size_);
    for (size_t i = 0; i < target_size_; ++i) {
        if (target_name != nullptr) {
            std::string w = target_name(i);
            fprintf(fp, "%s\t", w.c_str());
        } else {
            fprintf(fp, "%lu\t", i);
        }
        for (size_t j = 0; j < hidden_size_; ++j) {
            const char* tail = (j == hidden_size_ - 1) ? "\n" : " ";
            fprintf(fp, "%lf%s", target_hidden_[i * hidden_size_ + j], tail);
        }
    }
    fclose(fp);
}

template <typename T>
T BiWord2VecModel<T>::Predict(size_t source_id, size_t target_id) {
    T score = PredictRaw(source_id, target_id);
    return sigmoid_table_[score];
}

template <typename T>
T BiWord2VecModel<T>::PredictRaw(size_t source_id, size_t target_id) {
    if (source_id >= source_size_ || target_id >= target_size_) {
        return 0;
    }

    T sum = 0;
    for (size_t i = 0; i < hidden_size_; ++i) {
        sum +=
            source_hidden_[source_id * hidden_size_ + i] *
            target_hidden_[target_id * hidden_size_ + i];
    }

    return sum;
}

template <typename T>
T BiWord2VecModel<T>::Update(
    size_t source_id,
    size_t target_id,
    std::vector<size_t>& negative_targets,
    typename NoiseProbFunctionType<size_t, T>::Type noise_prob_func,
    T decay,
    T* buffer
) {
    bool delete_buffer = false;
    if (buffer == nullptr) {
        buffer = new T[hidden_size_ + 1];
        delete_buffer = true;
    }

    auto update_target = [&] (size_t source_id, size_t target_id, T grad) {
        for (size_t i = 0; i < hidden_size_; ++i) {
            buffer[i] -= grad * target_hidden_[target_id * hidden_size_ + i];
            target_hidden_[target_id * hidden_size_ + i] -=
                grad * source_hidden_[source_id * hidden_size_ + i];
        }
    };

    auto update_source = [&] (size_t source_id, T* buffer) {
        for (size_t i = 0; i < hidden_size_; ++i) {
            source_hidden_[source_id * hidden_size_ + i] += buffer[i];
        }
    };

    std::fill(buffer, buffer + hidden_size_, 0);

    T logloss = 0;
    // T pred = Predict(source_id, target_id);
    // update_target(source_id, target_id, alpha_ * decay * (pred - 1.));
    // logloss += -safe_log<T>(pred);
    T pred_raw = PredictRaw(source_id, target_id);
    if (noise_prob_func != nullptr) {
        pred_raw -= noise_prob_func(target_id);
    }
    T pred = sigmoid_table_[pred_raw];
    logloss += -sigmoid_table_.LogSigmoid(pred_raw);
    update_target(source_id, target_id, alpha_ * decay * (pred - 1.));

    for (size_t j = 0; j < negative_targets.size(); ++j) {
        size_t negative_id = negative_targets[j];
        // pred = Predict(source_id, negative_id);
        // update_target(source_id, negative_id, alpha_ * decay * pred);
        // logloss += -safe_log<T>(1. - pred);
        pred_raw = PredictRaw(source_id, negative_id);
        if (noise_prob_func != nullptr) {
            pred_raw -= noise_prob_func(negative_id);
        }
        pred = sigmoid_table_[pred_raw];
        logloss += -sigmoid_table_.LogSigmoid(-pred_raw);
        update_target(source_id, negative_id, alpha_ * decay * pred);
    }

    update_source(source_id, buffer);

    if (delete_buffer) {
        delete [] buffer;
    }

    return logloss;
}

template <typename IdType, typename T>
BiWord2VecTrainer<IdType, T>::BiWord2VecTrainer()
: data_manager_ {nullptr}, data_sampler_ {nullptr}, target_sampler_ {nullptr} {
    data_manager_ = new DataManager<IdType, T>();
}

template <typename IdType, typename T>
BiWord2VecTrainer<IdType, T>::~BiWord2VecTrainer() {
    if (data_manager_) {
        delete data_manager_;
    }

    if (data_sampler_) {
        delete data_sampler_;
    }

    if (target_sampler_) {
        delete target_sampler_;
    }
}

template <typename IdType, typename T>
bool BiWord2VecTrainer<IdType, T>::Train(
    const char* input_path,
    const char* model_path,
    T alpha,
    size_t hidden_size,
    size_t iteration,
    size_t negative,
    size_t training_words,
    size_t num_threads,
    double weight_neg_sampling,
    WeightType weight_type,
    LossType method,
    unsigned seed
) {
    data_manager_->load_data(input_path, num_threads);
    data_sampler_ = data_manager_->build_data_sampler(seed);
    target_sampler_ = data_manager_->build_target_sampler(seed, weight_neg_sampling, weight_type);

    BiWord2VecModel<T>* model = new BiWord2VecModel<T> (
        data_manager_->source_size(),
        data_manager_->target_size(),
        hidden_size,
        alpha
    );

    model->InitModel(seed);

    if (training_words == 0) {
        training_words = data_manager_->size();
    }

    std::unordered_map<IdType, T> target_unigram_prob;
    if (method == LOSS_NCE) {
        T total_weight = 0;
        for (size_t i = 0; i < data_manager_->size(); ++i) {
            const Sample<IdType, T>* sample = data_manager_->SampleAt(i);
            target_unigram_prob[sample->target()] += sample->weight();
            total_weight += sample->weight();
        }

        // for Noise-Constrastive Estimation: log(k * P_n(w))
        for (auto iter = target_unigram_prob.begin(); iter != target_unigram_prob.end(); ++iter) {
            iter->second = log(negative * iter->second / total_weight);
        }
    }

    TrainingContext* context = new TrainingContext();
    context->model = model;
    context->training_words = training_words;
    context->training_words_actual = 0;
    context->negative = negative;
    context->num_threads = num_threads;
    context->iteration = iteration;

    if (method == LOSS_NCE) {
        context->target_noise_prob = &target_unigram_prob;
    }

    num_threads = std::max(num_threads, static_cast<size_t>(1));

    std::thread *threads = new std::thread[num_threads];
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(
            &BiWord2VecTrainer<IdType, T>::TrainThread,
            this,
            i,
            context
        );
    }

    for (size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    double loss = context->logloss / context->logloss_count;
    printf("%cProgress: 100.00%%  Log-loss: %.4lf\n", 13, loss);

    auto source_name = [&] (size_t sid) {
        return data_manager_->SourceWord(sid);
    };

    auto target_name = [&] (size_t tid) {
        return data_manager_->TargetWord(tid);
    };

    model->Save(model_path, source_name, target_name);

    delete context;
    delete model;
    return true;
}

template <typename IdType, typename T>
void BiWord2VecTrainer<IdType, T>::TrainThread(
    size_t thread_id,
    TrainingContext* context
) {
    size_t hidden_size = context->model->hidden_size();
    T* buffer = new T[hidden_size + 1];

    size_t iteration = context->iteration;
    size_t num_threads = context->num_threads;
    size_t training_words = context->training_words;
    size_t local_training_words = (training_words * iteration + num_threads - 1) / num_threads;
    size_t negative = context->negative;

    typename NoiseProbFunctionType<size_t, T>::Type noise_prob_func = nullptr;
    if (context->target_noise_prob != nullptr) {
        noise_prob_func = [&] (size_t id) {
            auto iter = context->target_noise_prob->find(id);
            if (iter != context->target_noise_prob->end()) {
                return iter->second;
            }

            return T();
        };
    }

    size_t last_word_count = 0;
    T alpha_decay = 1;
    T logloss = 0;
    size_t count = 0;
    for (size_t i = 0; i < local_training_words; ++i) {
        size_t sample_id = data_sampler_->sampling();
        const Sample<IdType, T>* sample = data_manager_->SampleAt(sample_id);
        size_t source_id = sample->source();
        size_t target_id = sample->target();

        if (i - last_word_count > 10000 || i == local_training_words - 1) {
            context->logloss += logloss;
            context->logloss_count += count;
            context->training_words_actual += i - last_word_count;
            last_word_count = i;
            alpha_decay = 1. - context->training_words_actual * 1. /
                (context->training_words * iteration + 1.);
            alpha_decay = std::max(static_cast<T>(0.0001), alpha_decay);

            double progress = context->training_words_actual * 1. /
                (context->training_words * iteration);
            double loss = context->logloss / context->logloss_count;
            printf("%cProgress: %.2lf%%  Log-loss: %.4lf", 13, progress * 100, loss);
            fflush(stdout);
        }

        std::vector<size_t> negative_targets;
        for (size_t j = 0; j < negative; ++j) {
            size_t negative_id = target_sampler_->sampling();
            negative_targets.push_back(negative_id);
        }

        logloss += context->model->Update(
            source_id,
            target_id,
            negative_targets,
            noise_prob_func,
            alpha_decay,
            buffer
        );
        count += 1 + negative;
    }

    delete [] buffer;
}

#endif // SRC_BIWORD2VEC_H
/* vim: set ts=4 sw=4 tw=0 et :*/
