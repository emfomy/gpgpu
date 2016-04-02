#include "lab2.h"
#include "math.h"
static const unsigned kW = 960;
static const unsigned kH = 720;
static const unsigned kNFrame = 1080;
static const int kPixelDim = 2;
static const unsigned kFullW = kW*kPixelDim;
static const unsigned kFullH = kH*kPixelDim;
static const int kMaxIter = 1024;
static const int kColorIter = 32;
static const double kZoomBase = -M_LN2/24;
static const int kZoomIterStart = 24;
static const int kZoomIterEnd = 960;

// M(4, 1)
// static const double kCenterX = -0.10109636384562216103;
// static const double kCenterY = -0.95628651080914150077;

// M(4, 2)
// static const double kCenterX = -0.15248775721021457321;
// static const double kCenterY = -1.10344623936678258059;

//
// static const double kCenterX = -0.6367543465823900;
// static const double kCenterY =  0.6850312970836773;

//
static const double kCenterX =  0.001643721971153;
static const double kCenterY = -0.822467633298876;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The quadratic map, f(z) = z^2+c
///
__device__ void QuadraticMap( const double zx, const double zy, double &fx, double &fy, const double cx, const double cy ) {
    fx = (zx*zx - zy*zy) + cx;
    fy = 2*zx*zy + cy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Convert the iteration to YUV color
///
__device__ void Iter2YUV( const int iter, uint8_t &y, uint8_t &u, uint8_t &v ) {
  if ( iter == kMaxIter ) {
    y = 0;
    u = 128;
    v = 128;
  } else {
    double r, g, b, shift = (iter%kColorIter) * 255.0 / kColorIter;
    switch ( (iter/kColorIter) % 6 ) {
      case 0: { // b -> gb
        r = 0;
        g = shift;
        b = 255;
        break;
      }
      case 1: { // gb -> g
        r = 0;
        g = 255;
        b = 255 - shift;
        break;
      }
      case 2: { // g -> rg
        r = shift;
        g = 255;
        b = 0;
        break;
      }
      case 3: { // rg -> r
        r = 255;
        g = 255 - shift;
        b = 0;
        break;
      }
      case 4: { // r -> rb
        r = 255;
        g = 0;
        b = shift;
        break;
      }
      case 5: { // rb -> b
        r = 255 - shift;
        g = 0;
        b = 255;
        break;
      }
    }

    shift = (iter%(kColorIter/2)) * 0.5 / (kColorIter/2);
    if ( (iter/(kColorIter/2)) % 2 == 0 ) {
      r *= 1.0-shift;
      g *= 1.0-shift;
      b *= 1.0-shift;
    } else {
      r *= 0.5+shift;
      g *= 0.5+shift;
      b *= 0.5+shift;
    }

    y = 0.299*r + 0.587*g + 0.114*b;
    u =-0.169*r - 0.331*g + 0.500*b + 128;
    v = 0.500*r - 0.419*g - 0.081*b + 128;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Draw the Mandelbrot set
///
__global__
void MandelbrotSet( uint8_t (*colorY)[kW], uint8_t (*colorU)[kW/2], uint8_t (*colorV)[kW/2],
                    uint8_t (*colorFullY)[kFullW], uint8_t (*colorFullU)[kFullW], uint8_t (*colorFullV)[kFullW],
                    double zoom_iter ) {
  int idxx = blockIdx.x * blockDim.x + threadIdx.x;
  int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  int dimx = gridDim.x * blockDim.x;
  int dimy = gridDim.y * blockDim.y;
  double zoom = exp(zoom_iter * kZoomBase);

  // Compute iteration number
  double cx = double(idxx-dimx/2) / (64.0*kPixelDim) * zoom + kCenterX,
         cy = double(idxy-dimy/2) / (64.0*kPixelDim) * zoom + kCenterY,
         zx = 0, zy = 0, fx, fy;
  int iter;
  for ( iter = 0; (zx*zx+zy*zy) < 4.0 && iter < kMaxIter; ++iter ) {
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
  double zoom_iter = 0;
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
                                                impl->zoom_iter);
  if ( impl->t < kZoomIterStart ) {
    impl->zoom_iter += 1.0/(kZoomIterStart - impl->t);
  } else if ( impl->t > kZoomIterEnd ) {
    impl->zoom_iter += 1.0/(impl->t - kZoomIterEnd);
  } else {
    ++(impl->zoom_iter);
  }
  ++(impl->t);
}
