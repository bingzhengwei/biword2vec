#include "src/word_table.h"

WordTable::WordTable() {
    word_map_.reserve(DEFAULT_TABLE_SIZE);
}

WordTable::~WordTable() {
}

void WordTable::reserve(size_t table_size) {
    if (table_size > DEFAULT_TABLE_SIZE) {
        word_map_.reserve(table_size + 1);
    }
}

size_t WordTable::SearchWord(const std::string& word) {
    std::lock_guard<SpinLock> lock(lock_);
    auto iter = word_map_.find(word);
    if (iter == word_map_.end()) {
        // std::lock_guard<SpinLock> lock(lock_);
        iter = word_map_.find(word);
        if (iter == word_map_.end()) {
            size_t id = word_vec_.size();
            word_vec_.push_back(word);
            word_map_[word] = id;
            return id;
        } else {
            return iter->second;
        }
    } else {
        return iter->second;
    }
}

size_t WordTable::SearchWord(const char* word) {
    std::lock_guard<SpinLock> lock(lock_);
    auto iter = word_map_.find(word);
    if (iter == word_map_.end()) {
        // std::lock_guard<SpinLock> lock(lock_);
        iter = word_map_.find(word);
        if (iter == word_map_.end()) {
            size_t id = word_vec_.size();
            word_vec_.push_back(word);
            word_map_[word] = id;
            return id;
        } else {
            return iter->second;
        }
    } else {
        return iter->second;
    }
}

size_t WordTable::SearchWord(const std::string& word) const {
    auto iter = word_map_.find(word);
    if (iter != word_map_.end()) {
        return iter->second;
    } else {
        return this->npos;
    }
}

size_t WordTable::SearchWord(const char* word) const {
    auto iter = word_map_.find(word);
    if (iter != word_map_.end()) {
        return iter->second;
    } else {
        return this->npos;
    }
}

std::string WordTable::WordAt(size_t pos) {
    std::lock_guard<SpinLock> lock(lock_);
    if (pos >= word_vec_.size()) {
        return std::string();
    } else {
        return word_vec_[pos];
    }
}

/* vim: set ts=4 sw=4 tw=0 et :*/
