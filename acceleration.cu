#include <iostream>
#include <cuda_runtime.h>

__global__ void matMul(const int * A, const int * B, int * C, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row<N && col < N){
        int sum = 0;
        for(int j=0;j<N;j++){
            sum += A[row * N + j] * B[j * N + col];
        }
        C[row*N + col] = sum;
    }
}
extern "C" void launchMatMul(int* A, int* B, int* C, int N) {
    int * A_G, *B_G, *C_G;
    int size = sizeof(int) * N * N;
    cudaMalloc(&A_G, size);
    cudaMalloc(&B_G, size);
    cudaMalloc(&C_G, size);
    cudaMemcpy(A_G, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_G, A, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256; 

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    matMul<<<blocksPerGrid, threadsPerBlock>>>(A_G, B_G, C_G, N);
    cudaMemcpy(C, C_G, size, cudaMemcpyDeviceToHost);
    return;
}