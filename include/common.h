#pragma once

#include <stdint.h>

#include <algorithm>

#include "cublas_v2.h"
#include "cuda_fp16.h"

#include "mma.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32


using namespace nvcuda;


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr,                                                  \
                    "CUDA Error: %s:%d\n%s\n",                               \
                    __FILE__,                                                \
                    __LINE__,                                                \
                    cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

void fill_random(half* data, const int size)
{
    for (int i = 0; i < size; ++i) {
        data[i] =
            static_cast<half>(std::rand()) /
            static_cast<half>(RAND_MAX);
    }
}

__device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

