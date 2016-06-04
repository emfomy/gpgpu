__global__
void f1( float4* __restrict__ ptr ) {
  float4 v = ptr[threadIdx.x];
  v.x += 1;
  v.y += 1;
  v.z += 1;
  v.w += 1;
  ptr[threadIdx.x] = v;
}

__global__
void f2( float* __restrict__ ptr1, float* __restrict__ ptr2, float* __restrict__ ptr3, float* __restrict__ ptr4 ) {
  ptr1[threadIdx.x] += 1;
  ptr2[threadIdx.x] += 1;
  ptr3[threadIdx.x] += 1;
  ptr4[threadIdx.x] += 1;
}

int main() {
  float *some_ptr;
  cudaMalloc(&some_ptr, 128 * sizeof(float));
  f1<<<1, 32>>>((float4*) some_ptr);
  f2<<<1, 32>>>(some_ptr, some_ptr+32, some_ptr+64, some_ptr+96);
}
