#include <getopt.h>
#include <unistd.h>

#include "src/biword2vec.h"

typedef uint32_t id_t;
typedef float real_t;

void print_usage(int argc, char **argv) {
    printf("Usage: %s --input input_path --model model_path [options]\n"
        "options:\n"
        "--method LINE|NCE : set estimation method, default LINE\n"
        "--iter iteration : set number of iteration, default 1\n"
        "--alpha alpha : set learning rate, default 0.05\n"
        "--hidden hidden_size : set hidden size, default 10\n"
        "--negative negative : set count of negative sampling, default 5\n"
        "--words words : set number of words sampled per iteration\n"
        "--threads threads : set number of threads, default 1\n"
        "--weight_neg_sampling weight : set weight(exponential) "
        "for negative sampling, default 0\n"
        "--seed seed : set seed, default 1\n"
        "--help : print this help\n", argv[0]
    );
}

int main(int argc, char**argv) {
    int opt;
    int opt_idx = 0;

    static struct option long_options[] = {
        {"input", required_argument, nullptr, 'f'},
        {"model", required_argument, nullptr, 'm'},
        {"method", required_argument, nullptr, 'l'},
        {"iter", required_argument, nullptr, 'i'},
        {"alpha", required_argument, nullptr, 'a'},
        {"hidden", required_argument, nullptr, 'e'},
        {"negative", required_argument, nullptr, 'n'},
        {"words", required_argument, nullptr, 'w'},
        {"threads", required_argument, nullptr, 't'},
        {"weight_neg_sampling", required_argument, nullptr, 'p'},
        {"seed", required_argument, nullptr, 's'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    std::string input_path;
    std::string model_path;

    size_t iteration = 1;
    double alpha = 0.05;
    size_t hidden = 10;
    size_t negative = 5;
    size_t words_per_iter = 0;
    size_t threads = 1;
    unsigned seed = 1;
    double weight_neg_sampling = 0;
    LossType method = LOSS_LINE;

    while ((opt = getopt_long(argc, argv, "h", long_options, &opt_idx)) != -1) {
        switch (opt) {
        case 'f':
            input_path = optarg;
            break;
        case 'm':
            model_path = optarg;
            break;
        case 'l':
            if (!strcmp(optarg, "LINE")) {
                method = LOSS_LINE;
            } else if (!strcmp(optarg, "NCE")) {
                method = LOSS_NCE;
            } else {
                print_usage(argc, argv);
                exit(-1);
            }
            break;
        case 'i':
            iteration = static_cast<size_t>(atoi(optarg));
            break;
        case 'a':
            alpha = atof(optarg);
            break;
        case 'e':
            hidden = static_cast<size_t>(atoi(optarg));
            break;
        case 'n':
            negative = static_cast<size_t>(atoi(optarg));
            break;
        case 'p':
            weight_neg_sampling = atof(optarg);
            break;
        case 'w':
            words_per_iter = static_cast<size_t>(atoi(optarg));
            break;
        case 't':
            threads = static_cast<size_t>(atoi(optarg));
            break;
        case 's':
            seed = static_cast<unsigned>(atoi(optarg));
            break;
        case 'h':
        default:
            print_usage(argc, argv);
            exit(-1);
        }
    }

    if (input_path.size() == 0 || model_path.size() == 0) {
        print_usage(argc, argv);
        exit(-1);
    }

    BiWord2VecTrainer<id_t, real_t> trainer;

    trainer.Train(
        input_path.c_str(),
        model_path.c_str(),
        alpha,
        hidden,
        iteration,
        negative,
        words_per_iter,
        threads,
        weight_neg_sampling,
        method,
        seed
    );

    return 0;
}
/* vim: set ts=4 sw=4 tw=0 et :*/
