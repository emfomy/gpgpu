#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv( int a, int b ) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign( int a, int b ) { return CeilDiv(a, b) * b; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The kernel of Poisson image cloning
///
__global__ void PoissonImageCloningKernel(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox,
  const bool status
) {
  const int yt = blockIdx.y * blockDim.y + threadIdx.y;
  const int xt = blockIdx.x * blockDim.x + threadIdx.x;
  const int curt = wt*yt+xt;
  if ( (yt % 2) ^ (xt % 2) == status && 0 < yt && yt < ht-1 && 0 < xt && xt < wt-1 && mask[curt] > 127.0f ) {
    const int yb = oy+yt, xb = ox+xt;
    const int curb = wb*yb+xb;
    if ( 0 < yb && yb < hb-1 && 0 < xb && xb < wb-1 ) {
      output[curb*3+0] = target[curt*3+0]
                       +(output[(curb-wb)*3+0] + output[(curb-1)*3+0] + output[(curb+1)*3+0] + output[(curb+wb)*3+0]
                       - target[(curt-wt)*3+0] - target[(curt-1)*3+0] - target[(curt+1)*3+0] - target[(curt+wt)*3+0]) / 4;
      output[curb*3+1] = target[curt*3+1]
                       +(output[(curb-wb)*3+1] + output[(curb-1)*3+1] + output[(curb+1)*3+1] + output[(curb+wb)*3+1]
                       - target[(curt-wt)*3+1] - target[(curt-1)*3+1] - target[(curt+1)*3+1] - target[(curt+wt)*3+1]) / 4;
      output[curb*3+2] = target[curt*3+2]
                       +(output[(curb-wb)*3+2] + output[(curb-1)*3+2] + output[(curb+1)*3+2] + output[(curb+wb)*3+2]
                       - target[(curt-wt)*3+2] - target[(curt-1)*3+2] - target[(curt+1)*3+2] - target[(curt+wt)*3+2]) / 4;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The Poisson image cloning
///
void PoissonImageCloning(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
) {
  cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
  for ( auto i = 0; i < 20000; ++i ) {
    PoissonImageCloningKernel<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
      background, target, mask, output,
      wb, hb, wt, ht, oy, ox, true
    );
    PoissonImageCloningKernel<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
      background, target, mask, output,
      wb, hb, wt, ht, oy, ox, false
    );
  }
}
