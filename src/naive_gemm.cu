#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>


// ============================================================
// CUDA 错误检查宏
// ============================================================
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


// ============================================================
// Naive GEMM Kernel
//
// C = A * B
//
// A: M x K
// B: K x N
// C: M x N
//
// 每个 CUDA thread 负责计算 C 中的一个元素
// ============================================================
__global__ void naive_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ============================================================
// 初始化随机数据
// ============================================================
void fill_random(float* data, const int size)
{
    for (int i = 0; i < size; ++i) {
        data[i] =
            static_cast<float>(std::rand()) /
            static_cast<float>(RAND_MAX);
    }
}


// ============================================================
// 主函数
// ============================================================
int main()
{
    // -------------------------------
    // 矩阵尺寸
    // -------------------------------
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;


    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);


    // -------------------------------
    // Host 内存分配
    // -------------------------------
    float* h_A = static_cast<float*>(malloc(size_A));
    float* h_B = static_cast<float*>(malloc(size_B));
    float* h_C = static_cast<float*>(malloc(size_C));


    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }


    fill_random(h_A, M * K);
    fill_random(h_B, K * N);



    // -------------------------------
    // Device 内存分配
    // -------------------------------
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;


    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));


    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        size_A,
        cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        size_B,
        cudaMemcpyHostToDevice));



    // -------------------------------
    // Kernel 配置
    // -------------------------------
    constexpr int TILE_SIZE = 16;


    dim3 block(
        TILE_SIZE,
        TILE_SIZE);


    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE);



    // -------------------------------
    // Warmup
    // -------------------------------
    naive_gemm<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K);


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



    // -------------------------------
    // CUDA Event 计时
    // -------------------------------
    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));


    naive_gemm<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K);


    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));


    float elapsed_ms = 0.0f;


    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms,
        start,
        stop));



    // -------------------------------
    // 性能统计
    // -------------------------------
    const double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);


    const double gflops =
        flops /
        (elapsed_ms * 1e6);


    printf("--------------------------------\n");
    printf("Matrix size : %d x %d x %d\n",
           M,
           N,
           K);

    printf("Kernel time : %.3f ms\n",
           elapsed_ms);

    printf("Performance : %.2f GFLOPS\n",
           gflops);

    printf("--------------------------------\n");



    // -------------------------------
    // 拷贝结果回 Host
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        size_C,
        cudaMemcpyDeviceToHost));



    // -------------------------------
    // 资源释放
    // -------------------------------
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