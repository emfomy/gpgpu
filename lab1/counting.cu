#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

const int kNumThread = 1024;

__device__ __host__ int CeilDiv( int a, int b ) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign( int a, int b ) { return CeilDiv(a, b) * b; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Count position, step 1, device
///
__global__
void CountPositionDevice1( const char *text, int *pos, int text_size ) {
  auto idx = threadIdx.x;
  auto idx_shift = blockIdx.x * blockDim.x;
  text += idx_shift;
  pos  += idx_shift;

  // Initialize position
  if ( text[idx] == '\n' ) {
    pos[idx] = 0;
  } else if ( idx == 0 && idx_shift == 0 ) {
    pos[idx] = 1;
  } else {
    pos[idx] = -1;
  }
  __syncthreads();

  // Binary search
  for ( auto k = 1; k < blockDim.x; k *= 2 ) {
    auto idxk = idx + k;
    if ( idxk < blockDim.x && idxk+idx_shift < text_size && pos[idx] >= 0 && pos[idxk] < 0 ) {
      pos[idxk] = pos[idx] + k;
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Count position, step 2, device
///
__global__
void CountPositionDevice2( int *pos, int text_size ) {
  auto idx = threadIdx.x;
  pos += blockIdx.x * blockDim.x;

  if ( pos[idx] < 0 ) {
    pos[idx] = pos[-1] + idx + 1;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Count position, host
///
void CountPosition( const char *text, int *pos, int text_size ) {
  CountPositionDevice1<<<text_size / kNumThread + 1, kNumThread>>>(text, pos, text_size);
  cudaDeviceSynchronize();
  CountPositionDevice2<<<text_size / kNumThread + 1, kNumThread>>>(pos, text_size);
}

struct pred {
  __host__ __device__
  bool operator()( const int x ) {
    return (x == 1);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Extract head
///
int ExtractHead( const int *pos, int *head, int text_size) {
  int nhead;
  thrust::device_ptr<const int> pos_d(pos);
  thrust::device_ptr<int> head_d(head);

  // Extract head
  thrust::counting_iterator<int> idx_first(0), idx_last = idx_first + text_size;
  nhead = thrust::copy_if(idx_first, idx_last, pos_d, head_d, pred()) - head_d;

  return nhead;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Part 3, device, reverse words
///
__global__
void Part3Device( char *text, int *pos, int *head, int text_size, int n_head ) {
  auto i_word = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_end = (i_word == n_head-1) ? text_size : head[i_word+1];
  while ( pos[--idx_end] == 0 );

  char temp;
  for ( auto i = 0; i < (idx_end-head[i_word]) / 2; ++i ) {
    temp = text[head[i_word]+i];
    text[head[i_word]+i] = text[idx_end-i];
    text[idx_end-i] = temp;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Part 3
///
void Part3( char *text, int *pos, int *head, int text_size, int n_head ) {
// #pragma warning
//   const int kCheckSize = 1000;
//   char text_host[kCheckSize];

// #pragma warning
//   cudaMemcpy(text_host, text, sizeof(char) * kCheckSize, cudaMemcpyDeviceToHost);
//   for ( auto i = 0; i < kCheckSize; ++i ) {
//     if ( i % 100 == 0 ) {
//       printf("\n");
//     }
//     printf("%c", (text_host[i] == '\n') ? '_' : text_host[i]);
//   }
//   printf("\n");

  Part3Device<<<n_head / kNumThread + 1, kNumThread>>>(text, pos, head, text_size, n_head);

// #pragma warning
//   cudaMemcpy(text_host, text, sizeof(char) * kCheckSize, cudaMemcpyDeviceToHost);
//   for ( auto i = 0; i < kCheckSize; ++i ) {
//     if ( i % 100 == 0 ) {
//       printf("\n");
//     }
//     printf("%c", (text_host[i] == '\n') ? '_' : text_host[i]);
//   }
//   printf("\n");
}
