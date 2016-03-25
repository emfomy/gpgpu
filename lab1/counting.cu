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
__global__ void CountPositionDevice1( const char *text, int *pos, int text_size ) {
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
__global__ void CountPositionDevice2( int *pos, int text_size ) {
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Extract head
///
int ExtractHead( const int *pos, int *head, int text_size) {
  int *buffer;
  int nhead;
  cudaMalloc(&buffer, sizeof(int) * text_size * 2); // this is enough
  thrust::device_ptr<const int> pos_d(pos);
  thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

  // TODO
#pragma warning
  nhead = 0;

  cudaFree(buffer);
  return nhead;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Part 3
///
void Part3( char *text, int *pos, int *head, int text_size, int n_head ) {
#pragma warning
  static_cast<void>(text);
  static_cast<void>(pos);
  static_cast<void>(head);
  static_cast<void>(text_size);
  static_cast<void>(n_head);
}
