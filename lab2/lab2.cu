#include "lab2.h"
static const unsigned kW = 1280;
static const unsigned kH = 960;
static const unsigned kNFrame = 720;
static const int kPixelDim = 2;
static const unsigned kFullW = kW*kPixelDim;
static const unsigned kFullH = kH*kPixelDim;
static const int kMaxIter = 1024;
static const double kZoomBase = -0.693147180559945309417232121458/24;
static const double kCenterX = -0.77568377;
static const double kCenterY =  0.13646737;
// static const double kCenterX = -0.6367543465823900;
// static const double kCenterY =  0.6850312970836773;

__device__ void QuadraticMap( double zx, double zy, double &fx, double &fy, double cx, double cy ) {
    fx = (zx*zx - zy*zy) + cx;
    fy = 2*zx*zy + cy;
}

__device__ void Iter2YUV( int iter, uint8_t &y, uint8_t &u, uint8_t &v ) {
  double du, dv;
  if ( iter == kMaxIter ) {
    y = 0;
    u = 128;
    v = 128;
  } else {
    sincos(double(iter/8.0), &du, &dv);
    y = 128;
    u = -du*127 + 128;
    v =  dv*127 + 128;
  }
}

__global__
void MandelbrotSet( uint8_t (*colorY)[kW], uint8_t (*colorU)[kW/2], uint8_t (*colorV)[kW/2],
                    uint8_t (*colorFullY)[kFullW], uint8_t (*colorFullU)[kFullW], uint8_t (*colorFullV)[kFullW], int frame ) {
  int idxx = blockIdx.x * blockDim.x + threadIdx.x;
  int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  int dimx = gridDim.x * blockDim.x;
  int dimy = gridDim.y * blockDim.y;
  double zoom = exp(frame * kZoomBase);

  // Compute iteration number
  double cx = double(idxx-dimx/2) / (64.0*kPixelDim) * zoom + kCenterX,
         cy = double(idxy-dimy/2) / (64.0*kPixelDim) * zoom + kCenterY,
         zx = 0, zy = 0, fx, fy;
  int iter;
  for ( iter = 0; (zx*zx + zy*zy) < 4.0 && iter < kMaxIter; ++iter ) {
    QuadraticMap(zx, zy, fx, fy, cx, cy);
    zx = fx; zy = fy;
  }

  // Compute color
  Iter2YUV(iter, colorFullY[idxy][idxx], colorFullU[idxy][idxx], colorFullV[idxy][idxx]);
  __syncthreads();

  // Merge colors
  if ( threadIdx.x % kPixelDim == 0 && threadIdx.y % kPixelDim == 0 ) {
    int ytmp = 0;
    for ( auto j = 0; j < kPixelDim; ++j ) {
      for ( auto i = 0; i < kPixelDim; ++i ) {
        ytmp += colorFullY[idxy+j][idxx+i];
      }
    }
    colorY[idxy/kPixelDim][idxx/kPixelDim] = ytmp/(kPixelDim*kPixelDim);
  }
  if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
    int utmp = 0, vtmp = 0;
    for ( auto j = 0; j < 2*kPixelDim; ++j ) {
      for ( auto i = 0; i < 2*kPixelDim; ++i ) {
        utmp += colorFullU[idxy+j][idxx+i];
        vtmp += colorFullV[idxy+j][idxx+i];
      }
    }
    colorU[blockIdx.y][blockIdx.x] = utmp/(4*kPixelDim*kPixelDim);
    colorV[blockIdx.y][blockIdx.x] = vtmp/(4*kPixelDim*kPixelDim);
  }
}

struct Lab2VideoGenerator::Impl {
  int t = 0;
  uint8_t *colorFullY, *colorFullU, *colorFullV;
};

Lab2VideoGenerator::Lab2VideoGenerator() : impl( new Impl ) {
  cudaMalloc(&(impl->colorFullY), sizeof(uint8_t) * kFullW * kFullH);
  cudaMalloc(&(impl->colorFullU), sizeof(uint8_t) * kFullW * kFullH);
  cudaMalloc(&(impl->colorFullV), sizeof(uint8_t) * kFullW * kFullH);
}

Lab2VideoGenerator::~Lab2VideoGenerator() {
  cudaFree(impl->colorFullY);
  cudaFree(impl->colorFullU);
  cudaFree(impl->colorFullV);
}

void Lab2VideoGenerator::get_info( Lab2VideoInfo &info ) {
  info.w = kW;
  info.h = kH;
  info.n_frame = kNFrame;
  info.fps_n = 24;
  info.fps_d = 1;
};

void Lab2VideoGenerator::Generate( uint8_t *yuv ) {
  dim3 threadsPerBlock(2*kPixelDim, 2*kPixelDim);
  dim3 numBlocks(kW/2, kH/2);
  MandelbrotSet<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<uint8_t (*)[kW]>(yuv),
                                                reinterpret_cast<uint8_t (*)[kW/2]>(yuv+kW*kH),
                                                reinterpret_cast<uint8_t (*)[kW/2]>(yuv+kW*kH*5/4),
                                                reinterpret_cast<uint8_t (*)[kFullW]>(impl->colorFullY),
                                                reinterpret_cast<uint8_t (*)[kFullW]>(impl->colorFullU),
                                                reinterpret_cast<uint8_t (*)[kFullW]>(impl->colorFullV),
                                                impl->t);
  ++(impl->t);
}
