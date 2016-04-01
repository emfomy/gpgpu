#include "lab2.h"
static const unsigned W = 1280;
static const unsigned H = 960;
static const unsigned NFRAME = 1440;
static const int kMaxIter = 256;
// static const double kCenterX = -0.1528;
// static const double kCenterY =  1.0397;
static const double kCenterX = -0.6367543465823900;
static const double kCenterY =  0.6850312970836773;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
///
__global__
void MandelbrotSet( uint8_t Y[][W], uint8_t U[][W/2], uint8_t V[][W/2], int frame ) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int dimx = gridDim.x * blockDim.x;
  int dimy = gridDim.y * blockDim.y;
  double zoom = pow(1.0 + 1.0/24.0, double(frame));
  double cx = double(x-dimx/2) / 64.0 / zoom + kCenterX,
         cy = double(y-dimy/2) / 64.0 / zoom + kCenterY,
         zx = 0, zy = 0, wx, wy;
  int iter;
  for ( iter = 0; (zx*zx + zy*zy) < 4.0 && iter < kMaxIter; ++iter ) {
    wx = (zx*zx - zy*zy) + cx;
    wy = 2*zx*zy + cy;
    zx = wx; zy = wy;
  }
  if ( iter == kMaxIter ) {
    Y[2*y  ][2*x  ] = 0;
    Y[2*y  ][2*x+1] = 0;
    Y[2*y+1][2*x  ] = 0;
    Y[2*y+1][2*x+1] = 0;
    U[y][x] = 128;
    V[y][x] = 128;
  } else {
    Y[2*y  ][2*x  ] = 128;
    Y[2*y  ][2*x+1] = 128;
    Y[2*y+1][2*x  ] = 128;
    Y[2*y+1][2*x+1] = 128;
    double u, v;
    sincos(double(iter/8.0), &u, &v);
    U[y][x] =-u * 127+128;
    V[y][x] = v * 127+128;
  }
}

struct Lab2VideoGenerator::Impl {
  int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator() : impl( new Impl ) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info( Lab2VideoInfo &info ) {
  info.w = W;
  info.h = H;
  info.n_frame = NFRAME;
  info.fps_n = 24;
  info.fps_d = 1;
};

void Lab2VideoGenerator::Generate( uint8_t *yuv ) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(W / 2 / threadsPerBlock.x, H / 2 / threadsPerBlock.y);
  MandelbrotSet<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<uint8_t (*)[W]>(yuv),
                                                reinterpret_cast<uint8_t (*)[W/2]>(yuv+W*H),
                                                reinterpret_cast<uint8_t (*)[W/2]>(yuv+W*H*5/4),
                                                impl->t);
  ++(impl->t);
}
