#ifndef SRC_WORD_TABLE_H
#define SRC_WORD_TABLE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "src/lock.h"

const size_t DEFAULT_TABLE_SIZE = 30000000;

class WordTable {
public:
    WordTable();
    virtual ~WordTable();

    void reserve(size_t table_size = DEFAULT_TABLE_SIZE);

public:
    size_t SearchWord(const std::string& word);

    size_t SearchWord(const std::string& word) const;

    size_t SearchWord(const char* word);

    size_t SearchWord(const char* word) const;

    std::string WordAt(size_t pos);

    inline size_t size() {
        return word_vec_.size();
    }

public:
    static const size_t npos = -1;

private:
    std::unordered_map<std::string, size_t> word_map_;
    std::vector<std::string> word_vec_;
    SpinLock lock_;
};

#endif // SRC_WORD_TABLE_H
/* vim: set ts=4 sw=4 tw=0 et :*/
