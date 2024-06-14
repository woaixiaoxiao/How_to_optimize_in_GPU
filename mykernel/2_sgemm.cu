// optimize sgemm

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

constexpr int BUFFER_NUM = 2;
constexpr int THREAD_PER_BLOCK = 16 * 16;
constexpr int LOAD_UNIT = 4;
constexpr int THREADS_PER_WRAP = 32;
template <const int BLOCK_SIZE_M,    // height of block of C that each thread block calculate
          const int BLOCK_SIZE_K,    // width of block of A that each thread block load into shared
                                     // memory
          const int BLOCK_SIZE_N,    // width of block of C that each thread block calculate
          const int THREAD_SIZE_Y,   // height of block of C that each thread calculate
          const int THREAD_SIZE_X,   // width of block of C that each thread calculate
          const bool ENABLE_DOUBLE_BUFFER   // whether enable double buffering or not
          >
__global__ void Sgemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C,
                      const int M, const int N, const int K)
{
    // 1. 申请空间
    // 申请大tile使用的共享内存 A_SHARE B_SHARE
    __align__(16) __shared__ float AS[BUFFER_NUM][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __align__(16) __shared__ float BS[BUFFER_NUM][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // 申请小tile使用的寄存器 A_REGISTER B_REGISTER
    __align__(16) float AR[BUFFER_NUM][THREAD_SIZE_Y];
    __align__(16) float BR[BUFFER_NUM][THREAD_SIZE_X];
    // 申请空间用于矩阵A从全局内存load到共享内存中作为中转站，每个线程负责load一部分，因此只需要准备自己那部分的空间即可
    // 整个block对于A需要load BLOCK_SIZE_M×BLOCK_SIZE_K 个数据，总共有 blockDim.x × blockDim.y
    // 个线程， 因此每个线程需要load (BLOCK_SIZE_M×BLOCK_SIZE_K)/(blockDim.x × blockDim.y)
    constexpr int tempSize = (BLOCK_SIZE_M * BLOCK_SIZE_K) / (THREAD_PER_BLOCK);
    __align__(16) float ATEMP[tempSize];
    __align__(16) float BTEMP[tempSize];
    // 2.确定当前线程负责在矩阵AS和BS中load的位置
    // load时期线程的组织比较特别，将所有线程统一管理，每个线程依次读取4字节
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    // 确定矩阵A和B的共享矩阵中每一行需要多少个线程
    constexpr int THREADS_PER_ROW_AS = BLOCK_SIZE_K / LOAD_UNIT;
    constexpr int THREADS_PER_ROW_BS = BLOCK_SIZE_N / LOAD_UNIT;
    // 确定现在需要load AS和BS的哪一行，哪一列
    int LOAD_AS_ROW = tid / THREADS_PER_ROW_AS;
    int LOAD_AS_COL = (tid - LOAD_AS_ROW * THREADS_PER_ROW_AS) * LOAD_UNIT;
    int LOAD_AS_STRIDE = THREAD_PER_BLOCK / THREADS_PER_ROW_AS;
    int LOAD_BS_ROW = tid / THREADS_PER_ROW_BS;
    int LOAD_BS_COL = (tid - LOAD_BS_ROW * THREADS_PER_ROW_BS) * LOAD_UNIT;
    int LOAD_BS_STRIDE = THREAD_PER_BLOCK / THREADS_PER_ROW_BS;
    // 确定现在当前block在A中的起始行，B中的起始列
    int LOAD_A_START_ROW = blockIdx.y * BLOCK_SIZE_M;
    int LOAD_B_START_COL = blockIdx.x * BLOCK_SIZE_N;
    /*
    共享内存的形状是128×8，每个线程都需要读出一个8×8的小矩阵用于计算
    小矩阵后面的一个8就是128×8的后一个8，而前一个8则起始于(0,8,16...120)
    在用float4的方式读并且矩阵A已经被转置时，如果每个线程都读连续的8个数字，那么就会发生bankconflict（并不是说一个warp内的某两个线程访问一个bank就一定会发生conflict，因为存在一个warp内的所有线程都用float4的格式读数据
    在这种情况下，8个线程就读满了32个bank，共享内存的带宽已经被吃满了，也就不会有bankconflict带来的带宽不满）
    为了避免bankconflict，可以将每个线程要读的8个数据划分为两份4个数据，比如说第0个线程就访问0-4和64-67这8个数据
    那么如何划分呢
    */
    // 完成计算时下标的重映射
    int warpId = tid / THREADS_PER_WRAP;
    int laneId = tid % THREADS_PER_WRAP;
    int LOAD_FROM_AS = warpId / 2 * 16 + laneId / 8 * 4;
    int LOAD_FROM_BS = warpId % 2 * 32 + laneId % 8 * 4;
    constexpr int LOAD_FROM_S_STRIDE = 64;
    // 3.开始计算
    // 3.1 完成第一个大TILE的load，完成第一个小TILE的load
    int BIG_TILE_INDEX =
        0;   // 用于确定从全局内存load哪一块小矩阵到共享内存。通过该参数可以确定现在需要从全局内存load哪个128×8个矩阵到共享内存中
    int SMALL_TILE_INDEX = 0;   //
    //     用于确定从共享内存load哪一块小向量到寄存器。在现在的实现中，仅仅用于大tile的第一个小tile的搬运，不会被改变
    int S_WRITE_STAGE = 0;   // 对共享内存进行写操作的阶段，0或者1
    int R_WEITE_STAGE = 0;   //
    //     对寄存器进行写操作的阶段，0或者1。在现在的实现中，仅仅用于大tile的第一个小tile的搬运，不会被改变
    // 保存中间结果
    __align__(16) float tempSum[THREAD_SIZE_Y][THREAD_SIZE_X]{0.0f};
    // 从全局内存将下一个大tile读到共享内存中，会更新S_WRITE_STAGE
    auto LOAD_FROM_GLOBAL_TO_SHARE_1 = [&]() {
        _Pragma("unroll") for (size_t idx = 0; idx < BLOCK_SIZE_M; idx += LOAD_AS_STRIDE)
        {
            FETCH_FLOAT4(ATEMP[idx << 2]) = FETCH_FLOAT4(
                A[OFFSET(LOAD_A_START_ROW + LOAD_AS_ROW, LOAD_AS_COL + BIG_TILE_INDEX, K)]);
        }
        _Pragma("unroll") for (size_t idx = 0; idx < BLOCK_SIZE_K; idx += LOAD_BS_STRIDE)
        {
            FETCH_FLOAT4(BTEMP[idx << 2]) = FETCH_FLOAT4(
                B[OFFSET(LOAD_BS_ROW + BIG_TILE_INDEX, LOAD_BS_COL + LOAD_B_START_COL, N)]);
        }
    };
    auto LOAD_FROM_GLOBAL_TO_SHARE_2 = [&]() {
        _Pragma("unroll") for (size_t idx = 0; idx < BLOCK_SIZE_M; idx += LOAD_AS_STRIDE)
        {
            _Pragma("unroll") for (int j = 0; j < 4; j++)
            {
                AS[S_WRITE_STAGE][LOAD_AS_COL + j][LOAD_AS_ROW] = ATEMP[(idx << 2) + j];
            }
        }
        _Pragma("unroll") for (size_t idx = 0; idx < BLOCK_SIZE_K; idx += LOAD_BS_STRIDE)
        {
            FETCH_FLOAT4(BS[S_WRITE_STAGE][LOAD_BS_ROW][LOAD_BS_COL]) =
                FETCH_FLOAT4(BTEMP[idx << 2]);
        }
        __syncthreads();
        S_WRITE_STAGE ^= 1;
    };
    // 从共享内存中将下一个大tile的第一个小tile读到寄存器中，R_WEITE_STAGE固定为0，因为每个大tile的小tile数量是偶数，SMALL_TILE_INDEX固定为0，因为该函数只会作用于第一个小tile
    auto LOAD_FROM_SHARE_TO_REGISTER = [&]() {
        int S_LOAD_STAGE = S_WRITE_STAGE ^ 1;
        FETCH_FLOAT4(AR[R_WEITE_STAGE][0]) =
            FETCH_FLOAT4(AS[S_LOAD_STAGE][SMALL_TILE_INDEX][LOAD_FROM_AS]);
        FETCH_FLOAT4(AR[R_WEITE_STAGE][4]) =
            FETCH_FLOAT4(AS[S_LOAD_STAGE][SMALL_TILE_INDEX][LOAD_FROM_AS + LOAD_FROM_S_STRIDE]);
        FETCH_FLOAT4(BR[R_WEITE_STAGE][0]) =
            FETCH_FLOAT4(BS[S_LOAD_STAGE][SMALL_TILE_INDEX][LOAD_FROM_BS]);
        FETCH_FLOAT4(BR[R_WEITE_STAGE][4]) =
            FETCH_FLOAT4(BS[S_LOAD_STAGE][SMALL_TILE_INDEX][LOAD_FROM_BS + LOAD_FROM_S_STRIDE]);
    };
    // 一个小tile一个小tile的计算，每次计算前会先将下一个小tile加载进来
    auto COMPUTE_SMALL_TILE_BY_TILE = [&]() {
        _Pragma("unroll") for (int SMALL_TILE_INDEX = 1, R_WEITE_STAGE = 1;
                               SMALL_TILE_INDEX <= BLOCK_SIZE_K;
                               SMALL_TILE_INDEX += 1, R_WEITE_STAGE ^= 1)
        {
            if (SMALL_TILE_INDEX != BLOCK_SIZE_K) {
                LOAD_FROM_SHARE_TO_REGISTER();
            }
            int R_READ_STAGE = R_WEITE_STAGE ^ 1;
            _Pragma("unroll") for (size_t i = 0; i < THREAD_SIZE_Y; i++)
            {
                for (size_t j = 0; j < THREAD_SIZE_X; j++) {
                    tempSum[i][j] += AR[R_READ_STAGE][i] * BR[R_READ_STAGE][j];
                }
            }
        }
    };
// 3.2 循环遍历所有大TILE
#pragma unroll
    for (; BIG_TILE_INDEX <= K; BIG_TILE_INDEX += BLOCK_SIZE_K) {
        // BIG_TILE_INDEX=0时，需要读第0个大TILE
        if (BIG_TILE_INDEX == 0) {
            LOAD_FROM_GLOBAL_TO_SHARE_1();
            LOAD_FROM_GLOBAL_TO_SHARE_2();
            LOAD_FROM_SHARE_TO_REGISTER();
        }
        // BIG_TILE_INDEX=K时，需要计算最后一个大TILE，这个大TILE已经被读到了共享内存中
        else if (BIG_TILE_INDEX == K) {
            // 此时大TILE的第一个小TILE已经被读到了寄存器
            // 因此对于这个大TILE的计算，只需要读下一个小tile，计算当下的小tile
            COMPUTE_SMALL_TILE_BY_TILE();
        }
        // 其他情况下都是读第 BIG_TILE_INDEX 个大TILE，计算第 BIG_TILE_INDEX-1 个大TILE
        else {
            LOAD_FROM_GLOBAL_TO_SHARE_1();
            COMPUTE_SMALL_TILE_BY_TILE();
            LOAD_FROM_GLOBAL_TO_SHARE_2();
            LOAD_FROM_SHARE_TO_REGISTER();
        }
    }
    // 3.3 将结果写回到全局内存
    int GLOBAL_C_Y = blockIdx.y * BLOCK_SIZE_M;
    int GLOBAL_C_X = blockIdx.x * BLOCK_SIZE_N;
    int GLOBAL_C_Y_OFFSET[2]{LOAD_FROM_AS, LOAD_FROM_AS + LOAD_FROM_S_STRIDE};
    int GLOBAL_C_X_OFFSET[2]{LOAD_FROM_BS, LOAD_FROM_BS + LOAD_FROM_S_STRIDE};
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                FETCH_FLOAT4(C[OFFSET(
                    GLOBAL_C_Y + GLOBAL_C_Y_OFFSET[i] + k, GLOBAL_C_X + GLOBAL_C_X_OFFSET[j], N)]) =
                    FETCH_FLOAT4(tempSum[(i << 2) + k][(j << 2)]);
            }
        }
    }
}

void print44(float* a, int N)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.8f ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);
    // size_t M = LENGTH;
    // size_t K = LENGTH;
    // size_t N = LENGTH;

    assert(M % 8 == 0);
    assert(N % 8 == 0);
    assert(K % 8 == 0);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    // don't edit it
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    // 生成A的数据
    for (size_t i = 0; i < M * K; i++) {
        h_A[i] = i / 13;
    }

    // 生成B的数据
    for (size_t i = 0; i < K * N; i++) {
        h_B[i] = i % 13;
    }



    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M,
              BLOCK_SIZE_K,
              BLOCK_SIZE_N,
              THREAD_SIZE_Y,
              THREAD_SIZE_X,
              ENABLE_DOUBLE_BUFFER><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[0],
           msecPerMatrixMul[0],
           flopsPerMatrixMul);

    // cublas

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        cublasSgemm(
            blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[1],
           msecPerMatrixMul[1],
           flopsPerMatrixMul);

    cublasDestroy(blas_handle);


    double eps = 1.e-6;   // machine zero
    bool correct = true;

    for (size_t i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[i]);
        // double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%d][%d]=%.8f, ref=%.8f error term is > %E\n",
                   row,
                   col,
                   h_C[i],
                   h_C1[col * M + row],
                   eps);
            correct = false;
            if (i > 20) {
                break;
            }
        }
    }


    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}
