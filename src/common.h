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

inline size_t GetElementSize(nvinfer1::DataType t) noexcept {
    switch (t) {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

class HostMemory {
public:

    HostMemory() = delete;

    virtual void* Data() const noexcept {
        return m_data;
    }

    virtual std::size_t Size() const noexcept {
        return m_size;
    }
    virtual nvinfer1::DataType type() const noexcept {
        return m_type;
    }
    virtual ~HostMemory() {}

protected:

    HostMemory(std::size_t size, nvinfer1::DataType type)
        : m_size(size)
        , m_type(type) {
    }
    void* m_data{ nullptr };
    std::size_t m_size{ 0 };
    nvinfer1::DataType m_type;
};

template <typename ElemType, nvinfer1::DataType dataType>
class TypedHostMemory : public HostMemory {
public:
    explicit TypedHostMemory(std::size_t size)
        : HostMemory{ size, dataType } {
        this->m_data = new ElemType[size];
    };

    ~TypedHostMemory() noexcept {
        delete[](ElemType*) m_data;
    }

    ElemType *Raw() noexcept {
        return static_cast<ElemType *>(this->Data());
    }

    const ElemType *Raw() const noexcept {
        return static_cast<ElemType *>(this->Data());
    }
};

using FloatMemory = TypedHostMemory<float, nvinfer1::DataType::kFLOAT>;
using HalfMemory = TypedHostMemory<uint16_t, nvinfer1::DataType::kHALF>;
using ByteMemory = TypedHostMemory<uint8_t, nvinfer1::DataType::kINT8>;

template <typename T>
struct TrtxDestroyer {
    void operator()(T* t) const {
        t->destroy();
    }
};

template <typename T>
using TrtxUniquePtr = std::unique_ptr<T, TrtxDestroyer<T>>;

struct InferDeleter {
    template<typename T>
    void operator()(T *obj) const {
        if (nullptr != obj) {
            obj->destroy();
        }
    }
  };

template<typename T>
inline std::shared_ptr<T> InferObject(T *obj) {
    if (nullptr == obj) {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>{ obj, InferDeleter() };
}


}

#define GPU_ERROR_CHECK(code) {                     \
    trtx::GpuAssert((code), __FILE__, __LINE__);    \
}

#ifdef ASSERT 
#undef ASSERT
#endif

#define ASSERT(condition)                                                           \
    do {                                                                            \
        if (!(condition)) {                                                         \
            trtx::gLogError << "Assertion failure: " << #condition << std::endl;    \
            abort();                                                                \
        }                                                                           \
    } while (0)
