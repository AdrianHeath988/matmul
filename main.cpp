#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <cstdlib>
extern "C" void launchMatMul(int* A, int* B, int* C, int N);

//O(N^3) single thread
int naiveMatlul(int* A, int* B, int* C, int N){
    for(int i=0;i<N;i++){   //rows
        for(int j=0;j<N;j++){ // cols
            for(int k=0;k<N;k++){
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
    return 0;
}
//O(N^3) single thread but we do it i-k-j to make chache happy
int cachedNaiveMatMul(int* A, int* B, int* C, int N){
    for(int i=0;i<N;i++){  
        for(int k=0;k<N;k++){ 
            int tempA = A[i*N + k];
            #pragma GCC unroll 8
            for(int j=0;j<N;j++){
                C[i*N + j] += tempA * B[k*N + j];
            }
        }
    }
    return 0;
}
int cachedNaiveMatMulParallel(int* A, int* B, int* C, int N){
    #pragma omp parallel for 
    for(int i=0;i<N;i++){ 
        for(int k=0;k<N;k++){ 
            int tempA = A[i*N + k];
            #pragma GCC unroll 8
            for(int j=0;j<N;j++){
                C[i*N + j] += tempA * B[k*N + j];
            }
        }
    }
    return 0;
}
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
int cachedTiledMatMulParallel(int* __restrict A, int* __restrict B, int* __restrict C, int N, int BLOCK_SIZE) {


    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                
                // gotta account for the grunt of the litter
                int i_max = MIN(i0 + BLOCK_SIZE, N);
                int k_max = MIN(k0 + BLOCK_SIZE, N);
                // int j_max = MIN(j0 + BLOCK_SIZE, N);

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        int tempA = A[i*N + k];
                        
                        #pragma GCC ivdep
                        #pragma GCC unroll 8
                        for (int j = j0; j < j0 + BLOCK_SIZE; j++) {
                            C[i*N + j] += tempA * B[k*N + j];
                        }
                    }
                }
                
            }
        }
    }
    return 0;
}
int cachedTiledMatMulParallelAlligned(int* __restrict A_in, int* __restrict B_in, int* __restrict C_in, int N, int BLOCK_SIZE, int STRIDE) {

    int* __restrict A = (int*)__builtin_assume_aligned(A_in, 32);
    int* __restrict B = (int*)__builtin_assume_aligned(B_in, 32);
    int* __restrict C = (int*)__builtin_assume_aligned(C_in, 32);
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                
                // gotta account for the grunt of the litter
                int i_max = MIN(i0 + BLOCK_SIZE, N);
                int k_max = MIN(k0 + BLOCK_SIZE, N);
                // int j_max = MIN(j0 + BLOCK_SIZE, N);

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        int tempA = A[i*STRIDE + k];
                        
                        #pragma GCC ivdep
                        #pragma GCC unroll 8
                        for (int j = j0; j < j0 + BLOCK_SIZE; j++) {
                            C[i*STRIDE + j] += tempA * B[k*STRIDE + j];
                        }
                    }
                }
                
            }
        }
    }
    return 0;
}

// black magic time
int cachedTiledMatMulAVX(int* __restrict A_in, int* __restrict B_in, int* __restrict C_in, int N, int BLOCK_SIZE, int STRIDE) {
    
    int* __restrict A = (int*)__builtin_assume_aligned(A_in, 32);
    int* __restrict B = (int*)__builtin_assume_aligned(B_in, 32);
    int* __restrict C = (int*)__builtin_assume_aligned(C_in, 32);

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) { 
            for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                
                
                for (int i = i0; i < i0 + BLOCK_SIZE; i += 4) {
                    for (int j = j0; j < j0 + BLOCK_SIZE; j += 8) {
                        
                        
                        __m256i c0 = _mm256_load_si256((__m256i*)&C[(i+0)*STRIDE + j]);
                        __m256i c1 = _mm256_load_si256((__m256i*)&C[(i+1)*STRIDE + j]);
                        __m256i c2 = _mm256_load_si256((__m256i*)&C[(i+2)*STRIDE + j]);
                        __m256i c3 = _mm256_load_si256((__m256i*)&C[(i+3)*STRIDE + j]);
                        
                        for (int k = k0; k < k0 + BLOCK_SIZE; k++) {
                            __m256i b_vec = _mm256_load_si256((__m256i*)&B[k*STRIDE + j]);
                            
                            __m256i a0 = _mm256_set1_epi32(A[(i+0)*STRIDE + k]);
                            c0 = _mm256_add_epi32(c0, _mm256_mullo_epi32(a0, b_vec));
                            
                            __m256i a1 = _mm256_set1_epi32(A[(i+1)*STRIDE + k]);
                            c1 = _mm256_add_epi32(c1, _mm256_mullo_epi32(a1, b_vec));
                            
                            __m256i a2 = _mm256_set1_epi32(A[(i+2)*STRIDE + k]);
                            c2 = _mm256_add_epi32(c2, _mm256_mullo_epi32(a2, b_vec));
                            
                            __m256i a3 = _mm256_set1_epi32(A[(i+3)*STRIDE + k]);
                            c3 = _mm256_add_epi32(c3, _mm256_mullo_epi32(a3, b_vec));
                        }
                        
                        _mm256_store_si256((__m256i*)&C[(i+0)*STRIDE + j], c0);
                        _mm256_store_si256((__m256i*)&C[(i+1)*STRIDE + j], c1);
                        _mm256_store_si256((__m256i*)&C[(i+2)*STRIDE + j], c2);
                        _mm256_store_si256((__m256i*)&C[(i+3)*STRIDE + j], c3);
                    }
                }
                
            }
        }
    }
    return 0;
}
//WE HATE THE OS FUCK WSL
// I HATE THE TLB I HATE THE TLB I HATE THE TLB
int cachedTiledMatMulAVX_Ultimate(int* __restrict A_in, int* __restrict B_in, int* __restrict C_in, int N, int BLOCK_SIZE, int STRIDE) {
    
    int* __restrict A = (int*)__builtin_assume_aligned(A_in, 32);
    int* __restrict B = (int*)__builtin_assume_aligned(B_in, 32);
    int* __restrict C = (int*)__builtin_assume_aligned(C_in, 32);

    
    int* B_pack = (int*)aligned_alloc(32, N * N * sizeof(int));
    
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; j += 8) {
        for (int k = 0; k < N; k++) {
            __m256i b_vec = _mm256_load_si256((__m256i*)&B[k*STRIDE + j]);
            // j * N jumps to the panel, k * 8 jumps down the rows sequentially
            _mm256_store_si256((__m256i*)&B_pack[j * N + k * 8], b_vec);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) { 
            for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                
                for (int i = i0; i < i0 + BLOCK_SIZE; i += 4) {
                    for (int j = j0; j < j0 + BLOCK_SIZE; j += 8) {
                        
                        __m256i c0_A = _mm256_load_si256((__m256i*)&C[(i+0)*STRIDE + j]);
                        __m256i c1_A = _mm256_load_si256((__m256i*)&C[(i+1)*STRIDE + j]);
                        __m256i c2_A = _mm256_load_si256((__m256i*)&C[(i+2)*STRIDE + j]);
                        __m256i c3_A = _mm256_load_si256((__m256i*)&C[(i+3)*STRIDE + j]);
                        
                        __m256i c0_B = _mm256_setzero_si256();
                        __m256i c1_B = _mm256_setzero_si256();
                        __m256i c2_B = _mm256_setzero_si256();
                        __m256i c3_B = _mm256_setzero_si256();
                        
                        int b_ptr = j * N + k0 * 8;
                        
                        for (int k = k0; k < k0 + BLOCK_SIZE; k += 2) {
                            
                            __m256i b_vec_0 = _mm256_load_si256((__m256i*)&B_pack[b_ptr]);
                            
                            c0_A = _mm256_add_epi32(c0_A, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+0)*STRIDE + k]), b_vec_0));
                            c1_A = _mm256_add_epi32(c1_A, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+1)*STRIDE + k]), b_vec_0));
                            c2_A = _mm256_add_epi32(c2_A, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+2)*STRIDE + k]), b_vec_0));
                            c3_A = _mm256_add_epi32(c3_A, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+3)*STRIDE + k]), b_vec_0));
                            
                            __m256i b_vec_1 = _mm256_load_si256((__m256i*)&B_pack[b_ptr + 8]);
                            
                            c0_B = _mm256_add_epi32(c0_B, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+0)*STRIDE + k + 1]), b_vec_1));
                            c1_B = _mm256_add_epi32(c1_B, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+1)*STRIDE + k + 1]), b_vec_1));
                            c2_B = _mm256_add_epi32(c2_B, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+2)*STRIDE + k + 1]), b_vec_1));
                            c3_B = _mm256_add_epi32(c3_B, _mm256_mullo_epi32(_mm256_set1_epi32(A[(i+3)*STRIDE + k + 1]), b_vec_1));
                            
                            b_ptr += 16;
                        }
                        
                        c0_A = _mm256_add_epi32(c0_A, c0_B);
                        c1_A = _mm256_add_epi32(c1_A, c1_B);
                        c2_A = _mm256_add_epi32(c2_A, c2_B);
                        c3_A = _mm256_add_epi32(c3_A, c3_B);
                        
                        _mm256_store_si256((__m256i*)&C[(i+0)*STRIDE + j], c0_A);
                        _mm256_store_si256((__m256i*)&C[(i+1)*STRIDE + j], c1_A);
                        _mm256_store_si256((__m256i*)&C[(i+2)*STRIDE + j], c2_A);
                        _mm256_store_si256((__m256i*)&C[(i+3)*STRIDE + j], c3_A);
                    }
                }
                
            }
        }
    }
    
    free(B_pack);
    return 0;
}


int main() {
    // Program which generates two matricies of size NxN, then multiplies using cuda
    int N = 1<<13;
    int STRIDE = N + 32;
    long N_3 = N*N;
    N_3 = N_3 * N;
    double m_ops = N_3/1000000;
    std::cout << "Doing " << m_ops << " million total operations." << std::endl;
    //init prng
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(1, 10);

    //generate rand matricies:
    // int * A = (int *)malloc(sizeof(int) * N * N);
    // int * B = (int *)malloc(sizeof(int) * N * N);
    // int * C = (int *)malloc(sizeof(int) * N * N);
    int* A = (int*)aligned_alloc(32, N * STRIDE * sizeof(int));
    int* B = (int*)aligned_alloc(32, N * STRIDE * sizeof(int));
    int* C = (int*)aligned_alloc(32, N * STRIDE * sizeof(int));
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for N = %d\n", N);
        exit(EXIT_FAILURE);
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            A[i * N + j] = distrib(gen);
            B[i * N + j] = distrib(gen);
            
        }
    }
    auto start = std::chrono::steady_clock::now();
    // naiveMatlul(A, B, C, N); //86.3325 million operations per second aka 0.086 GIOPS
    // cachedNaiveMatMul(A, B, C, N); //5.1 GIOPS
    // cachedNaiveMatMulParallel(A, B, C, N); // 14.4 GIOPS
    // cachedTiledMatMulParallel(A, B, C, N, 32); //37.2 GIOPS
    // cachedTiledMatMulParallelAlligned(A, B, C, N, 32, STRIDE); // 43 GIOPS
    // cachedTiledMatMulAVX(A, B, C, N, 32, STRIDE); //74 GIOPS
    cachedTiledMatMulAVX_Ultimate(A, B, C, N, 32, STRIDE); //118 GIOPS (95% of theoretical max)
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed.count() << " ms\n";
    std::cout << m_ops/(elapsed.count() /1000.0) << " million operations per second." <<std::endl;
    // printf("A matrix:\n");
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         printf("%f, ", A[i * N + j]);
            
    //     }
    //     printf("\n");
    // }
    // printf("B matrix:\n");
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         printf("%f, ", B[i * N + j]);
            
    //     }
    //     printf("\n");
    // }
    // printf("C matrix:\n");
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         printf("%f, ", C[i * N + j]);
            
    //     }
    //     printf("\n");
    // }

    printf("Complete.");
    return 0;

}