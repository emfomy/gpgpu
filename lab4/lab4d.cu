static const int DIM = 128;

__global__
void Normalize128( float *data, int N ) {
  float tmp[DIM];
  float norm1 = 0;
  float *start = data + threadIdx.x + blockIdx.x*blockDim.x;
#pragma unroll
  for ( int i = 0; i < DIM; ++i ) {
    tmp[i] = *start;
    norm1 += abs(tmp[i]);
    start += N;
  }
  float norm1_inv = 1.0f / norm1;
  start = data + threadIdx.x + blockIdx.x*blockDim.x;
#pragma unroll
  for ( int i = 0; i < DIM; ++i ) {
    *start = tmp[i] * norm1_inv;
    start += N;
  }
}
