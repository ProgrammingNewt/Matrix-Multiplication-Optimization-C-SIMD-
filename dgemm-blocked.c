#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Blocked dgemm, column-major, AVX.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * do_block:
 *   Computes a sub-block of C += (sub-block of A) * (sub-block of B).
 * 
 *   All pointers (A_block, B_block, C_block) point to the "top-left corner"
 *   of their respective sub-blocks in column-major order. Dimensions:
 *     - M rows, K columns in A_sub
 *     - K rows, N columns in B_sub
 *     - M rows, N columns in C_sub
 *
 *   n is the leading dimension of the *full* matrices.
 *
 *   We use an AVX micro-kernel that processes 8 rows of A (and C) at a time.
 */


static void micro_kernel_4_1(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c0 = _mm256_loadu_pd(C_block);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal = _mm256_broadcast_sd(B_block + p);

        // c0..c1 += a0..a1 * bVal
        c0 = _mm256_fmadd_pd(a0, bVal, c0);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c0);
}

static void micro_kernel_8_1(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c0 = _mm256_loadu_pd(C_block);
    __m256d c1 = _mm256_loadu_pd(C_block + 4);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);
        __m256d a1 = _mm256_loadu_pd(A_block + 4 + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal = _mm256_broadcast_sd(B_block + p);

        // c0..c1 += a0..a1 * bVal
        c0 = _mm256_fmadd_pd(a0, bVal, c0);
        c1 = _mm256_fmadd_pd(a1, bVal, c1);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c0);
    _mm256_storeu_pd(C_block + 4, c1);
}

static void micro_kernel_16_1(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c0 = _mm256_loadu_pd(C_block);
    __m256d c1 = _mm256_loadu_pd(C_block + 4);
    __m256d c2 = _mm256_loadu_pd(C_block + 8);
    __m256d c3 = _mm256_loadu_pd(C_block + 12);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);
        __m256d a1 = _mm256_loadu_pd(A_block + 4 + p*M);
        __m256d a2 = _mm256_loadu_pd(A_block + 8 + p*M);
        __m256d a3 = _mm256_loadu_pd(A_block + 12 + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal = _mm256_broadcast_sd(B_block + p);

        // c0..c1 += a0..a1 * bVal
        c0 = _mm256_fmadd_pd(a0, bVal, c0);
        c1 = _mm256_fmadd_pd(a1, bVal, c1);
        c2 = _mm256_fmadd_pd(a2, bVal, c2);
        c3 = _mm256_fmadd_pd(a3, bVal, c3);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c0);
    _mm256_storeu_pd(C_block + 4, c1);
    _mm256_storeu_pd(C_block + 8, c2);
    _mm256_storeu_pd(C_block + 12, c3);
}



static void micro_kernel_4_2(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c00 = _mm256_loadu_pd(C_block);
    __m256d c01 = _mm256_loadu_pd(C_block + n);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        // __m256d a0 = _mm256_loadu_pd(A_block + i2 + p*n);
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal0 = _mm256_broadcast_sd(B_block + p);
        __m256d bVal1 = _mm256_broadcast_sd(B_block + p + n);

        // c0..c1 += a0..a1 * bVal
        c00 = _mm256_fmadd_pd(a0, bVal0, c00);
        c01 = _mm256_fmadd_pd(a0, bVal1, c01);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c00);
    _mm256_storeu_pd(C_block + n, c01);
}


static void micro_kernel_8_2(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c00 = _mm256_loadu_pd(C_block);
    __m256d c10 = _mm256_loadu_pd(C_block + 4);
    __m256d c01 = _mm256_loadu_pd(C_block + n);
    __m256d c11 = _mm256_loadu_pd(C_block + 4 + n);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);
        __m256d a1 = _mm256_loadu_pd(A_block + 4 + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal0 = _mm256_broadcast_sd(B_block + p);
        __m256d bVal1 = _mm256_broadcast_sd(B_block + p + n);

        // c0..c1 += a0..a1 * bVal
        c00 = _mm256_fmadd_pd(a0, bVal0, c00);
        c10 = _mm256_fmadd_pd(a1, bVal0, c10);
        c01 = _mm256_fmadd_pd(a0, bVal1, c01);
        c11 = _mm256_fmadd_pd(a1, bVal1, c11);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c00);
    _mm256_storeu_pd(C_block + 4, c10);
    _mm256_storeu_pd(C_block + n, c01);
    _mm256_storeu_pd(C_block + 4 + n, c11);
}

static void micro_kernel_16_2(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c00 = _mm256_loadu_pd(C_block);
    __m256d c10 = _mm256_loadu_pd(C_block + 4);
    __m256d c20 = _mm256_loadu_pd(C_block + 8);
    __m256d c30 = _mm256_loadu_pd(C_block + 12);
    __m256d c01 = _mm256_loadu_pd(C_block + n);
    __m256d c11 = _mm256_loadu_pd(C_block + 4 + n);
    __m256d c21 = _mm256_loadu_pd(C_block + 8 + n);
    __m256d c31 = _mm256_loadu_pd(C_block + 12 + n);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);
        __m256d a1 = _mm256_loadu_pd(A_block + 4 + p*M);
        __m256d a2 = _mm256_loadu_pd(A_block + 8 + p*M);
        __m256d a3 = _mm256_loadu_pd(A_block + 12 + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal0 = _mm256_broadcast_sd(B_block + p);
        __m256d bVal1 = _mm256_broadcast_sd(B_block + p + n);

        // c0..c1 += a0..a1 * bVal
        c00 = _mm256_fmadd_pd(a0, bVal0, c00);
        c10 = _mm256_fmadd_pd(a1, bVal0, c10);
        c20 = _mm256_fmadd_pd(a2, bVal0, c20);
        c30 = _mm256_fmadd_pd(a3, bVal0, c30);
        c01 = _mm256_fmadd_pd(a0, bVal1, c01);
        c11 = _mm256_fmadd_pd(a1, bVal1, c11);
        c21 = _mm256_fmadd_pd(a2, bVal1, c21);
        c31 = _mm256_fmadd_pd(a3, bVal1, c31);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c00);
    _mm256_storeu_pd(C_block + 4, c10);
    _mm256_storeu_pd(C_block + 8, c20);
    _mm256_storeu_pd(C_block + 12, c30);
    _mm256_storeu_pd(C_block + n, c01);
    _mm256_storeu_pd(C_block + 4 + n, c11);
    _mm256_storeu_pd(C_block + 8 + n, c21);
    _mm256_storeu_pd(C_block + 12 + n, c31);
}



static void micro_kernel_4_6(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c0 = _mm256_loadu_pd(C_block);
    __m256d c1 = _mm256_loadu_pd(C_block + n);
    __m256d c2 = _mm256_loadu_pd(C_block + 2*n);
    __m256d c3 = _mm256_loadu_pd(C_block + 3*n);
    __m256d c4 = _mm256_loadu_pd(C_block + 4*n);
    __m256d c5 = _mm256_loadu_pd(C_block + 5*n);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal0 = _mm256_broadcast_sd(B_block + p);
        __m256d bVal1 = _mm256_broadcast_sd(B_block + p + n);
        __m256d bVal2 = _mm256_broadcast_sd(B_block + p + 2*n);
        __m256d bVal3 = _mm256_broadcast_sd(B_block + p + 3*n);
        __m256d bVal4 = _mm256_broadcast_sd(B_block + p + 4*n);
        __m256d bVal5 = _mm256_broadcast_sd(B_block + p + 5*n);

        // c0..c1 += a0..a1 * bVal
        c0 = _mm256_fmadd_pd(a0, bVal0, c0);
        c1 = _mm256_fmadd_pd(a0, bVal1, c1);
        c2 = _mm256_fmadd_pd(a0, bVal2, c2);
        c3 = _mm256_fmadd_pd(a0, bVal3, c3);
        c4 = _mm256_fmadd_pd(a0, bVal4, c4);
        c5 = _mm256_fmadd_pd(a0, bVal5, c5);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c0);
    _mm256_storeu_pd(C_block + n, c1);
    _mm256_storeu_pd(C_block + 2*n, c2);
    _mm256_storeu_pd(C_block + 3*n, c3);
    _mm256_storeu_pd(C_block + 4*n, c4);
    _mm256_storeu_pd(C_block + 5*n, c5);
}



static void micro_kernel_8_6(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Load the 8 current C values into two AVX registers
    // (column-major means i2..i2+7 are consecutive in memory).
    __m256d c00 = _mm256_loadu_pd(C_block);
    __m256d c10 = _mm256_loadu_pd(C_block + 4);
    __m256d c01 = _mm256_loadu_pd(C_block + n);
    __m256d c11 = _mm256_loadu_pd(C_block + 4 + n);
    __m256d c02 = _mm256_loadu_pd(C_block + 2*n);
    __m256d c12 = _mm256_loadu_pd(C_block + 4 + 2*n);
    __m256d c03 = _mm256_loadu_pd(C_block + 3*n);
    __m256d c13 = _mm256_loadu_pd(C_block + 4 + 3*n);
    __m256d c04 = _mm256_loadu_pd(C_block + 4*n);
    __m256d c14 = _mm256_loadu_pd(C_block + 4 + 4*n);
    __m256d c05 = _mm256_loadu_pd(C_block + 5*n);
    __m256d c15 = _mm256_loadu_pd(C_block + 4 + 5*n);

    // Accumulate over the K dimension
    for (int p = 0; p < K; p++)
    {
        // A_block(i2..i2+7, p) => load in two 4-double chunks
        __m256d a0 = _mm256_loadu_pd(A_block + p*M);
        __m256d a1 = _mm256_loadu_pd(A_block + 4 + p*M);

        // B_block(p, j2) is a single double => broadcast
        __m256d bVal0 = _mm256_broadcast_sd(B_block + p);
        __m256d bVal1 = _mm256_broadcast_sd(B_block + p + n);

        // c0..c1 += a0..a1 * bVal
        c00 = _mm256_fmadd_pd(a0, bVal0, c00);
        c10 = _mm256_fmadd_pd(a1, bVal0, c10);
        c01 = _mm256_fmadd_pd(a0, bVal1, c01);
        c11 = _mm256_fmadd_pd(a1, bVal1, c11);

        // B_block(p, j2) is a single double => broadcast
        bVal0 = _mm256_broadcast_sd(B_block + p + 2*n);
        bVal1 = _mm256_broadcast_sd(B_block + p + 3*n);

        // c0..c1 += a0..a1 * bVal
        c02 = _mm256_fmadd_pd(a0, bVal0, c02);
        c12 = _mm256_fmadd_pd(a1, bVal0, c12);
        c03 = _mm256_fmadd_pd(a0, bVal1, c03);
        c13 = _mm256_fmadd_pd(a1, bVal1, c13);

        // B_block(p, j2) is a single double => broadcast
        bVal0 = _mm256_broadcast_sd(B_block + p + 4*n);
        bVal1 = _mm256_broadcast_sd(B_block + p + 5*n);

        // c0..c1 += a0..a1 * bVal
        c04 = _mm256_fmadd_pd(a0, bVal0, c04);
        c14 = _mm256_fmadd_pd(a1, bVal0, c14);
        c05 = _mm256_fmadd_pd(a0, bVal1, c05);
        c15 = _mm256_fmadd_pd(a1, bVal1, c15);
    }

    // Store updated results back to C
    _mm256_storeu_pd(C_block, c00);
    _mm256_storeu_pd(C_block + 4, c10);
    _mm256_storeu_pd(C_block + n, c01);
    _mm256_storeu_pd(C_block + 4 + n, c11);
    _mm256_storeu_pd(C_block + 2*n, c02);
    _mm256_storeu_pd(C_block + 4 + 2*n, c12);
    _mm256_storeu_pd(C_block + 3*n, c03);
    _mm256_storeu_pd(C_block + 4 + 3*n, c13);
    _mm256_storeu_pd(C_block + 4*n, c04);
    _mm256_storeu_pd(C_block + 4 + 4*n, c14);
    _mm256_storeu_pd(C_block + 5*n, c05);
    _mm256_storeu_pd(C_block + 4 + 5*n, c15);
}

// Aim for 8 x 6 micro-kernel
// Do a 16 x 2 micro-kernel
static void do_block(int n, int M, int K, int N,
                     double* A_block,    // pointer to A-subblock (col-major)
                     double* B_block,    // pointer to B-subblock (col-major)
                     double* C_block)    // pointer to C-subblock (col-major)
{
    // Loop over columns of the sub-block of C: j2 in [0..N-1]
    for (int j2 = 0; j2 < 6*(N/6); j2 += 6)
    {
        // Loop over rows of the sub-block of C in steps of 8
        for (int i2 = 0; i2 < 8*(M/8); i2 += 8)
        {
            micro_kernel_8_6(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }
        
        // A final block of 4
        if (8*(M/8) != 4*(M/4))
        {
            int i2 = 8*(M/8);
            micro_kernel_4_6(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);   
        }

        // Remainder loop for rows if M is not multiple of 4
        for (int i2 = 4*(M/4); i2 < M; i2++)
        {
            double sum0 = C_block[i2 + j2*n];
            double sum1 = C_block[i2 + (j2+1)*n];
            double sum2 = C_block[i2 + (j2+2)*n];
            double sum3 = C_block[i2 + (j2+3)*n];
            double sum4 = C_block[i2 + (j2+4)*n];
            double sum5 = C_block[i2 + (j2+5)*n];
            for (int p = 0; p < K; p++)
            {
                sum0 += A_block[i2 + p*M] * B_block[p + j2*n];
                sum1 += A_block[i2 + p*M] * B_block[p + (j2+1)*n];
                sum2 += A_block[i2 + p*M] * B_block[p + (j2+2)*n];
                sum3 += A_block[i2 + p*M] * B_block[p + (j2+3)*n];
                sum4 += A_block[i2 + p*M] * B_block[p + (j2+4)*n];
                sum5 += A_block[i2 + p*M] * B_block[p + (j2+5)*n];
            }
            C_block[i2 + j2*n] = sum0;
            C_block[i2 + (j2+1)*n] = sum1;
            C_block[i2 + (j2+2)*n] = sum2;
            C_block[i2 + (j2+3)*n] = sum3;
            C_block[i2 + (j2+4)*n] = sum4;
            C_block[i2 + (j2+5)*n] = sum5;
        }
    }

    // Resolve the last few columns
    for (int j2 = 6*(N/6); j2 < 2*(N/2); j2 += 2)
    {
        // Loop over rows of the sub-block of C in steps of 8
        for (int i2 = 0; i2 < 16*(M/16); i2 += 16)
        {
            micro_kernel_16_2(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }

        // A final block of 8
        if (16*(M/16) != 8*(M/8))
        {

            int i2 = 16*(M/16);
            micro_kernel_8_2(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }
        
        // A final block of 4
        if (8*(M/8) != 4*(M/4))
        {

            int i2 = 8*(M/8);
            micro_kernel_4_2(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }


        // Remainder loop for rows if M is not multiple of 4
        for (int i2 = 4*(M/4); i2 < M; i2++)
        {
            double sum0 = C_block[i2 + j2*n];
            double sum1 = C_block[i2 + (j2+1)*n];
            for (int p = 0; p < K; p++)
            {
                sum0 += A_block[i2 + p*M] * B_block[p + j2*n];
                sum1 += A_block[i2 + p*M] * B_block[p + (j2+1)*n];
            }
            C_block[i2 + j2*n] = sum0;
            C_block[i2 + (j2+1)*n] = sum1;
        }
    }

    if (2*(N/2) != N) 
    {
        int j2 = 2*(N/2);
        // Loop over rows of the sub-block of C in steps of 8
        for (int i2 = 0; i2 < 16*(M/16); i2 += 16)
        {
            micro_kernel_16_1(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }

        // A final block of 8
        if (16*(M/16) != 8*(M/8))
        {

            int i2 = 16*(M/16);
            micro_kernel_8_1(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }
        
        // A final block of 4
        if (8*(M/8) != 4*(M/4))
        {

            int i2 = 8*(M/8);
            micro_kernel_4_1(n, M, K, N, A_block + i2, B_block + j2*n, C_block + i2+j2*n);
        }


        // Remainder loop for rows if M is not multiple of 4
        for (int i2 = 4*(M/4); i2 < M; i2++)
        {
            double sum = C_block[i2 + j2*n];
            for (int p = 0; p < K; p++)
            {
                // sum += A_block[i2 + p*n] * B_block[p + j2*n];
                sum += A_block[i2 + p*M] * B_block[p + j2*n];
            }
            C_block[i2 + j2*n] = sum;
        }
    }
}

/*
 * square_dgemm:
 *   - Blocks the matrices in 3 nested loops of block size BLOCK_SIZE
 *   - Calls do_block on each sub-block
 *
 *   A, B, C are assumed to be n-by-n in column-major layout, i.e.:
 *     A(i, j) is at A[i + j*n]
 */
void square_dgemm(int n, double* A, double* B, double* C)
{

    double *new_A = malloc(n*n*sizeof(double));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            int q = i/BLOCK_SIZE;
            int r = i%BLOCK_SIZE;
            int new_index = BLOCK_SIZE*q*n + r + min(BLOCK_SIZE, n - BLOCK_SIZE*q)*j;
            new_A[new_index] = A[i+n*j];
        }
    }
    // new_A stores entries of A as 00, 10, ... , (BLOCK_SIZE-1)0, 01, 11, ... , (BLOCK_SIZE-1)1, 02, 12, ... , (BLOCK_SIZE-1)2, and so on. 
    // That way, one block of A is contiguous.

    // 3-level blocking
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                //Compute the actual block sizes near edges
                int M = min(BLOCK_SIZE, n - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, n - k);

                //Top-left corners of sub-blocks in column-major
                // double* A_block = A + i + k*n;  //sub-block of A
                double* A_block = new_A + n*i + M*k;
                double* B_block = B + k + j*n;  // sub-block of B
                double* C_block = C + i + j*n;  //sub-block of C

                // Multiply & accumulate the sub-block
                do_block(n, M, K, N, A_block, B_block, C_block);
            }
        }
    }
    free(new_A);
}
