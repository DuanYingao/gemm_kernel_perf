#include "common.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(expr)                                                   \
    do {                                                                     \
        cublasStatus_t _s = (expr);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                    \
                    (int)_s, __FILE__, __LINE__);                            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(){
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    const size_t size_A = M * K * sizeof(half);
    const size_t size_B = K * N * sizeof(half);
    const size_t size_C = M * N * sizeof(half);

    // Host 内存
    half* h_A = static_cast<half*>(malloc(size_A));
    half* h_B = static_cast<half*>(malloc(size_B));
    half* h_C = static_cast<half*>(malloc(size_C));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    // Device 内存
    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // ---------------- cuBLAS 初始化（只一次） ----------------
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    // CUDA 13 中 CUBLAS_TENSOR_OP_MATH 已废弃，配合 CUBLAS_COMPUTE_16F 无需再 setMathMode
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    half alpha = __float2half(1.0f);
    half beta  = __float2half(0.0f);

    // ---------------- 计时 ----------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // A: row-major M*K, B: col-major K*N, C: row-major M*N
    // column-major cuBLAS 算 row-major C=A*B：交换 A/B 与 M/N
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, K,
        d_A, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // ---------------- 性能统计 ----------------
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = flops / (elapsed_ms * 1e6);

    printf("--------------------------------\n");
    printf("Matrix size : %d x %d x %d\n", M, N, K);
    printf("Kernel time : %.3f ms\n", elapsed_ms);
    printf("Performance : %.2f GFLOPS\n", gflops);
    printf("--------------------------------\n");

    // 结果拷回
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // 释放
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}