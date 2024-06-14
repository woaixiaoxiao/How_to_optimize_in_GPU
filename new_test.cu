#include <iostream>
#include <cuda_runtime.h>

__global__ void test()
{
    int tid = threadIdx.x;
    auto test_lambda = [&] __device__() { printf("%d\n", tid); };
    test_lambda();
}

int main()
{
    test<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}