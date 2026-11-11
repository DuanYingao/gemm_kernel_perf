#include "common.h"

__global__ void naive_wmma(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K * N + warp_col, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    naive_wmma<<<grid, block>>>(A, B, C, M, N, K);
}

int main(){
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    const size_t size_A = M * K * sizeof(half);
    const size_t size_B = K * N * sizeof(half);
    const size_t size_C = M * N * sizeof(half);


    // -------------------------------
    // Host 内存分配
    // -------------------------------
    half* h_A = static_cast<half*>(malloc(size_A));
    half* h_B = static_cast<half*>(malloc(size_B));
    half* h_C = static_cast<half*>(malloc(size_C));


    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }


    fill_random(h_A, M * K);
    fill_random(h_B, K * N);



    // -------------------------------
    // Device 内存分配
    // -------------------------------
    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C = nullptr;

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
    
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
    

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));


    naive_wmma<<<grid, block>>>(
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