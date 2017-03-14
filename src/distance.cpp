#include <getopt.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

#include "src/util.h"

typedef std::function<double(double*, double*, size_t n)> score_func_t;
typedef std::pair<std::string, double> pair_t;

enum SearchSpace { SPACE_SOURCE = 0, SPACE_TARGET = 1, SPACE_ALIGNMENT = 2 };

double dot(double* v1, double* v2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

double l2norm(double* v1, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += v1[i] * v1[i];
    }

    return sqrt(sum);
}

double cosine(double* v1, double* v2, size_t n) {
    double numerator = dot(v1, v2, n);
    double denominator = l2norm(v1, n) * l2norm(v2, n);
    if (util_equal<double> (denominator, 0)) {
        return 0;
    }

    return numerator / denominator;
}

class EmbeddingModel {
public:
    EmbeddingModel() : num_feat_ {0}, hidden_size_ {0}, model_ {nullptr}, init_ {false} {}

    virtual ~EmbeddingModel() {
        if (model_ != nullptr) {
            delete [] model_;
        }
        init_ = false;
    }

    bool LoadModel(const char* path) {
        FILE* file_desc = fopen(path, "r");

        if (!file_desc) {
            return false;
        }

        enum { BUF_SIZE = 102400 };
        char* buf = new char[BUF_SIZE];

        size_t line_num = 0;
        bool error_flag = false;
        char *ptr = nullptr;
        while (true) {
            if (fgets(buf, BUF_SIZE - 1, file_desc) == nullptr) {
                break;
            }
            buf[BUF_SIZE - 1] = '\0';

            if (line_num == 0) {
                if (sscanf(buf, "%lu %lu", &num_feat_, &hidden_size_) != 2) {
                    error_flag = true;
                    break;
                }
                model_ = new double[num_feat_ * hidden_size_ + 1];
                std::fill(model_, model_ + num_feat_ * hidden_size_ + 1, 0);
            } else if (line_num <= num_feat_) {
                char *word = strtok_r(buf, "\t\r\n", &ptr);
                if (word == nullptr) {
                    error_flag = true;
                    break;
                }
                size_t feat_id = line_num - 1;
                id_map_[word] = feat_id;
                feat_name_.push_back(word);
                for (size_t i = 0; i < hidden_size_; ++i) {
                    char *p = strtok_r(NULL, " \t\r\n", &ptr);
                    if (p == nullptr) {
                        break;
                    }
                    model_[feat_id * hidden_size_ + i] = atof(p);
                }
            } else {
                break;
            }
            ++line_num;
        }

        delete [] buf;
        fclose(file_desc);

        if (!error_flag || line_num <= num_feat_ || feat_name_.size() != num_feat_) {
            init_ = true;
        }
        return init_;
    }

    double* Embedding(const std::string& word) {
        auto iter = id_map_.find(word);
        if (iter == id_map_.end()) {
            return nullptr;
        }

        size_t feat_id = iter->second;
        if (feat_id >= num_feat_) {
            return nullptr;
        }

        return model_ + feat_id * hidden_size_;
    }

    std::vector<pair_t> Match(
        double* embedding,
        score_func_t score_func = cosine,
        size_t topn = 20
    ) {
        auto cmp = [] (const pair_t& left, const pair_t& right) {
            return left.second > right.second;
        };

        std::priority_queue<pair_t, std::vector<pair_t>, decltype(cmp)> q(cmp);
        for (size_t i = 0; i < num_feat_; ++i) {
            double* embedding_i = model_ + i * hidden_size_;
            double score = score_func(embedding, embedding_i, hidden_size_);
            q.push(pair_t(feat_name_[i], score));
            while (q.size() > topn) {
                q.pop();
            }
        }

        std::vector<pair_t> vec;
        while (q.size() > 0) {
            vec.push_back(q.top());
            q.pop();
        }

        std::sort(vec.begin(), vec.end(), cmp);

        return vec;
    }

protected:
    std::unordered_map<std::string, size_t> id_map_;
    std::vector<std::string> feat_name_;
    size_t num_feat_;
    size_t hidden_size_;
    double* model_;
    bool init_;
};

void print_usage(int argc, char **argv) {
    printf("Usage: %s --model model_path [options]\n"
        "options:\n"
        "--topn n : set number of results returned, default 20\n"
        "--space source|target|alignment : set search method, default alignment\n"
        "--score-func cosine|dot : set scoring function, default cosine\n"
        "--help : print this help\n", argv[0]
    );
}

int main(int argc, char* argv[]) {
    int opt;
    int opt_idx = 0;

    static struct option long_options[] = {
        {"model", required_argument, nullptr, 'm'},
        {"topn", required_argument, nullptr, 'n'},
        {"score-func", required_argument, nullptr, 's'},
        {"space", required_argument, nullptr, 'x'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    std::string model_path;
    size_t topn = 20;
    score_func_t score_func = cosine;
    SearchSpace space = SPACE_ALIGNMENT;

    while ((opt = getopt_long(argc, argv, "h", long_options, &opt_idx)) != -1) {
        switch (opt) {
        case 'm':
            model_path = optarg;
            break;
        case 'n':
            topn = static_cast<size_t>(atoi(optarg));
            break;
        case 'x':
            if (!strcmp(optarg, "source")) {
                space = SPACE_SOURCE;
            } else if (!strcmp(optarg, "target")) {
                space = SPACE_TARGET;
            } else if (!strcmp(optarg, "alignment")) {
                space = SPACE_ALIGNMENT;
            } else {
                print_usage(argc, argv);
                exit(-1);
            }
            break;
        case 's':
            if (!strcmp(optarg, "cosine")) {
                score_func = cosine;
            } else if (!strcmp(optarg, "dot")) {
                score_func = dot;
            } else {
                print_usage(argc, argv);
                exit(-1);
            }
            break;
        case 'h':
        default:
            print_usage(argc, argv);
            exit(-1);
        }
    }

    if (model_path.size() == 0) {
        print_usage(argc, argv);
        exit(-1);
    }

    std::string source_path = model_path + std::string(".source");
    std::string target_path = model_path + std::string(".target");

    EmbeddingModel model_source, model_target;

    if (space == SPACE_SOURCE) {
        model_source.LoadModel(source_path.c_str());
        model_target.LoadModel(source_path.c_str());
    } else if (space == SPACE_TARGET) {
        model_source.LoadModel(target_path.c_str());
        model_target.LoadModel(target_path.c_str());
    } else {
        model_source.LoadModel(source_path.c_str());
        model_target.LoadModel(target_path.c_str());
    }

    std::string word;
    std::cout << "Please Input:" << std::flush;
    while (std::cin >> word) {
        double* source_embedding = model_source.Embedding(word);
        if (source_embedding == nullptr) {
            std::cout << std::endl << word << " do not exist!" << std::endl;
            std::cout << "Please Input:" << std::flush;
            continue;
        }

        auto res = model_target.Match(source_embedding, score_func, topn);
        std::cout << std::endl;
        for (size_t i = 0; i < res.size(); ++i) {
            std::cout << res[i].first << "\t" << res[i].second << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Please Input:" << std::flush;
    }

    return 0;
}
