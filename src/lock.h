#ifndef SRC_LOCK_H
#define SRC_LOCK_H

#include <atomic>
#include <mutex>

class SpinLock {
public:
    SpinLock() {
    }

    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }


protected:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

#endif // SRC_LOCK_H
/* vim: set ts=4 sw=4 tw=0 et :*/
