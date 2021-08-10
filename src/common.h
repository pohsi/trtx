#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include "logger.h"

namespace trtx {

inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (cudaSuccess != code) {
        gLogError << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
        if (true == abort) {
            ::exit(code);
        }
    }
}

#define GPU_ERROR_CHECK(code) {             \
    GpuAssert((code), __FILE__, __LINE__);  \
}

template <typename T>
struct TrtxDestroyer {
    void operator()(T* t) const {
        t->destroy();
    }
};

template <typename T>
using TrtxUniquePtr = std::unique_ptr<T, TrtxDestroyer<T>>;

}