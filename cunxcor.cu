#if 0
#define _FERMI_
#endif

#if 0
#define _DEBUG_
#endif

#include <cassert>
#include <cfloat>
#include <cstdio>
#include <vector>

#define   NGRIDMAX 65535
#ifdef _FERMI_
#define NTHREADMAX 1024
#else
#define NTHREADMAX 512
#endif

#define WARP_SIZE2 5
#define WARP_SIZE (1<<WARP_SIZE2)

#define __out
template<const int SIZE2>
inline int ALIGN(const int n) {return ((n-1)&(-(1<<SIZE2))) + (1<<SIZE2);}

#ifndef _FERMI_
__forceinline__ __device__ int __mul(const int a, const int b) { return __mul24(a,b); }
#else
__forceinline__ __device__ int __mul(const int a, const int b) { return a*b; }
#endif


__constant__ int nImages;

__constant__ int imageNX;
__constant__ int imageNY;
__constant__ int imageSize;

__constant__ int   templateNX;
__constant__ int   templateNY;
__constant__ int   templateSize;
__constant__ float templateCoeff;

__constant__ int resultNX;
__constant__ int resultNY;
__constant__ int resultSize;


/* intra-block reduction */

  template<const int NTHREADS>
__device__ float reduce(float sum, volatile float *shmem)
{
  const int tid = threadIdx.x;
  shmem[tid] = sum;
  __syncthreads();

  /* reduce in shmem */
  if (NTHREADS >=1024) { if (tid < 512) { shmem[tid] = sum = sum + shmem[tid + 512]; } __syncthreads(); }
  if (NTHREADS >= 512) { if (tid < 256) { shmem[tid] = sum = sum + shmem[tid + 256]; } __syncthreads(); }
  if (NTHREADS >= 256) { if (tid < 128) { shmem[tid] = sum = sum + shmem[tid + 128]; } __syncthreads(); }
  if (NTHREADS >= 128) { if (tid <  64) { shmem[tid] = sum = sum + shmem[tid +  64]; } __syncthreads(); }
  if (tid < 32)
  {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    shmem[tid] = sum = sum + shmem[tid + 32];
    shmem[tid] = sum = sum + shmem[tid + 16]; 
    shmem[tid] = sum = sum + shmem[tid +  8];
    shmem[tid] = sum = sum + shmem[tid +  4];
    shmem[tid] = sum = sum + shmem[tid +  2];
    shmem[tid] = sum = sum + shmem[tid +  1]; 
  }

  __syncthreads();
  return shmem[0];
}

/* intra-block inclusive prefix sum */

template<int NTHREADS2>
__device__ void inclusive_prefix_sum(float sum, volatile float *shmem)
{
  const int tid = threadIdx.x;
  shmem[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int i = 0; i < NTHREADS2; i++)
  {
    const int offset = 1 << i;
    if (tid >= offset) sum += shmem[tid - offset]; 
    __syncthreads();
    shmem[tid] = sum;
    __syncthreads();
  }
}

/********************************/
/********************************/
/********************************/

/* subtracts mean value from the template */
template<const int NTHREADS>
__global__ void dev_computeTemplate(float *templateInOut)
{
  __shared__ float shmem[NTHREADS];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (bid >= nImages) return;
  
  const int       imageIdx = bid;
  const int templateOffset = imageIdx * templateSize;
  __out float   *templateD = templateInOut + templateOffset;
  
  float templateSum  = 0.0f;
  for (int i = 0; i < templateSize; i += NTHREADS)
    if (i + tid < templateSize)
      templateSum += templateD[i + tid];
  templateSum = reduce<NTHREADS>(templateSum, shmem);

  const float templateMean = templateSum * templateCoeff;

  for (int i = 0; i < templateSize; i += NTHREADS)
    if (i + tid < templateSize)
      templateD[i + tid] -= templateMean;
}  /* works */

/********************************/
/********************************/
/********************************/

/* normalizes correlation matrix : 
 *   C(u,v) -> C(u,v)/(\sum_{xy}(I_{x+u,y+v} - \bar{I}_{u,v})^2 \sum_{x,y}\hat{T}_{x,y}^2)^{0.5}
 *-----------
 * This version only works with images with imageNX <= NTHREADSMAX
 * where NTHREADSMAX is the maximal #threads/block 
 */

template<const int NTHREADS2>
__device__ float2 partialSums(const float v, volatile float* shmem)
{
  const int tid = threadIdx.x;

  volatile float *shMem  = shmem + 1;
  volatile float *shMem2 = shMem + 1 + (1 << NTHREADS2);

  inclusive_prefix_sum<NTHREADS2>(v,   shMem);
  inclusive_prefix_sum<NTHREADS2>(v*v, shMem2);
  const float Sum  = shMem [tid-1 + templateNX] - shMem [tid-1];
  const float Sum2 = shMem2[tid-1 + templateNX] - shMem2[tid-1];
  __syncthreads();

  return make_float2(Sum, Sum2);
} 

template<const int NTHREADS2>
__global__ void dev_normalizeCorr(
    const float *imageIn,
    const float *templateIn,
    __out float *resultOut)
{
  const int NTHREADS = 1<<NTHREADS2;
  __shared__ float shmem[NTHREADS*3];

  const int tid = threadIdx.x;
  const int bid =  blockIdx.x;
 
  if (bid >= nImages) return;

  const int       imageIdx = bid;
  const int    imageOffset = imageIdx *    imageSize;
  const int templateOffset = imageIdx * templateSize;
  const int   resultOffset = imageIdx *   resultSize;

  const float *   imageD =    imageIn  +    imageOffset;
  const float *templateD = templateIn  + templateOffset;
  __out float *  resultD =   resultOut +   resultOffset;

  /**********/

  float templateSum2 = 0.0f;
  for (int i = 0; i < templateSize; i += NTHREADS)
    if (i + tid < templateSize)
    {
      const float t = templateD[i + tid];
      templateSum2 += t*t;
    }
  templateSum2 = reduce<NTHREADS>(templateSum2, shmem);
  __syncthreads();

  /*********/

  shmem[tid] = shmem[tid + NTHREADS] = 0.0f;
  __syncthreads();

  float imageSum  = 0.0f;
  float imageSum2 = 0.0f;
  int iaddr = 0;
  const int windowSize = templateNY*imageNX;
  while (iaddr < windowSize)
  {
    const float2 res = partialSums<NTHREADS2>(imageD[iaddr + tid], shmem);
    imageSum  += res.x;
    imageSum2 += res.y;
    iaddr     += imageNX;
  }

  if (tid < resultNX)
  {
    const float norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
    resultD[tid] *= rsqrtf(norm2 + FLT_EPSILON);
  } 
  
  /*********/

  while (iaddr < imageSize)
  {
    const float2 res1 = partialSums<NTHREADS2>(imageD[iaddr-windowSize + tid], shmem);
    const float2 res2 = partialSums<NTHREADS2>(imageD[iaddr            + tid], shmem);
    imageSum  += res2.x - res1.x;
    imageSum2 += res2.y - res1.y;
    iaddr     += imageNX;

    if (tid < resultNX)
    {
      const int         iy = iaddr/imageNX;
      const int       addr = __mul(iy-templateNY, resultNX);
      const float    norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
      resultD[addr + tid] *= rsqrtf(norm2 + FLT_EPSILON);
    }
  }
}  /* works */

/********************************/
/********************************/
/********************************/

/* computes correlation matrix:
 *  C(u,v) = \sum_{xy} I(x+u, y+v) T(x,y) 
 *    !!! Here in stead you can also use cuFFT, should be faster !!!
 *-----------
 * This version only works with images with imageNX <= NTHREADSMAX
 * where NTHREADSMAX is the maximal #threads/block 
 */

template<const int NTHREADS, const int NPT>
__global__ void dev_nxcor(
    const float *imageIn,
    const float *templateIn,
    __out float *resultOut)
{
  __shared__ float shmem[NTHREADS*(1+NPT)];
  const int tid = threadIdx.x;
  const int bid =  blockIdx.x;
  const int  yc =  blockIdx.y*NPT;

  const int       imageIdx = bid;
  const int    imageOffset = imageIdx *    imageSize;
  const int templateOffset = imageIdx * templateSize;
  const int   resultOffset = imageIdx *   resultSize;

  const float *   imageD =    imageIn  +    imageOffset + tid;
  const float *templateD = templateIn  + templateOffset + tid;
  __out float *  resultD =   resultOut +   resultOffset;

  const int q  = min(NTHREADS/resultNX, 4);
  const int nt = NTHREADS/q;
  const int ty = threadIdx.x / nt;
  const int tx = threadIdx.x - nt * ty;

  const int templateNXq = templateNX/q;
  const int jbeg = templateNXq * ty;
  const int jend = ty+1 >= q ? templateNX : templateNXq + jbeg;

  float *shTemplate = shmem;
  float *shImage    = shmem + NTHREADS;
  float *shImage1   = shImage + tx;

  float corrCoeff[NPT];
  for (int k = 0; k < NPT; k++)
    corrCoeff[k] = 0.0f;

  int iaddr = yc*imageNX;

  /* this is the main NT*MT loop: 
   *   processing NPT elements per thread
   *   on GT200, use NPT = 3
   *   on FERMI, use NPT = 8
   *       with 8 elements/thread on FERMI (GTX580)
   *       the perfomance increase is up to 2.3x compared to GT200 with NPT=3
   */

  float img[NPT];
  for (int k = 0; k < NPT-1; k++, iaddr += imageNX)
    img[k] = imageD[iaddr]; 
  for (int taddr = 0; taddr < templateSize; taddr += templateNX, iaddr += imageNX)
  {
    shTemplate[tid] = templateD[taddr];
    img     [NPT-1] =    imageD[iaddr];
    for (int k = 0; k < NPT; k++)
      shImage[tid + NTHREADS*k] = img[k];
    for (int k = 0; k < NPT-1; k++)
      img[k] = img[k+1];
    __syncthreads();

    if (tx < resultNX && ty < q)
    {
#pragma unroll 8  /* unroll 8 seems to be most optimal */
      for (int j = jbeg; j < jend; j++)
        for (int k = 0; k < NPT; k++)
          corrCoeff[k] += shTemplate[j]*shImage1[j + NTHREADS*k];
    }
    __syncthreads();
  }

  for (int k = 0; k < NPT; k++)
    shmem[tid + NTHREADS*k] = corrCoeff[k];
  __syncthreads();

  for (int j = tx + nt; j < NTHREADS; j += nt)
    for (int k = 0; k < NPT; k++)
      corrCoeff[k] += shmem[j + NTHREADS*k];
  __syncthreads();

  if (tid < resultNX)
  {
    int raddr = yc*resultNX + tid;
    for (int k = 0; k < NPT; k++, raddr += resultNX)
      if (raddr < resultSize)
        resultD[raddr] = corrCoeff[k];
  }
}

/* matrix transpose */
template<const int TILE>
__global__ void dev_transpose(
    __out float *dstIn,
    const float *srcIn,
    const   int  nx,
    const   int  ny,
    const   int  gridDim_x)
{
  __shared__ float tile[TILE][TILE+1];

  const int tx  = threadIdx.x;
  const int ty  = threadIdx.y;
  const int bid =  blockIdx.x;
  const int Idx =  blockIdx.y;

  const   int size = Idx*nx*ny;
  const float *src = srcIn + size;
  __out float *dst = dstIn + size;

  const int blockIdx_y = bid / gridDim_x;
  const int blockIdx_x = bid - blockIdx_y * gridDim_x;

  const int bx = blockIdx_x;
  const int by = blockIdx_y;

  const int src_x   = bx*TILE + tx;
  const int src_y   = by*TILE + ty;
  const int src_idx = src_x + src_y * nx;

  const int dst_x   = by*TILE + tx;
  const int dst_y   = bx*TILE + ty;
  const int dst_idx = dst_x + dst_y * ny;


  if (src_x < nx && src_y < ny)
    tile[ty][tx] = src[src_idx];

  __syncthreads();

  if (dst_x < ny && dst_y < nx)
    dst[dst_idx] = tile[tx][ty];
}




#ifdef __cplusplus
extern "C"
#endif
void cunxcor_main(
    const float *   imageH,
    const float *templateH,
    __out float *  resultH,
    const int      nImages,
    int      imageNY,
    int      imageNX,
    int   templateNY,
    int   templateNX,
    const bool transpose)
{
  /* computing auxiliary variables */
  assert(   imageNX > 0);
  assert(   imageNY > 0);
  assert(templateNX > 0);
  assert(templateNY > 0);

  assert(nImages <    NGRIDMAX);
  assert(imageNX <= NTHREADMAX);

  int resultNX = imageNX - templateNX + 1;
  int resultNY = imageNY - templateNY + 1;
  assert(resultNX > 0);
  assert(resultNY > 0);

  const int    imageSize =    imageNX *    imageNY;
  const int templateSize = templateNX * templateNY;
  const int   resultSize =   resultNX *   resultNY;

  const float templateCoeff = 1.0f/(float)templateSize;

  /* allocating DRAM */

#ifdef _DEBUG_
  fprintf(stderr, " -- Allocating DRAM -- \n");
#endif

  int err = 0;
  float *imageD, *templateD, *resultD;
  const int       fpSize = sizeof(float);
  err = cudaMalloc(&   imageD, 2*    imageSize*nImages*fpSize); assert(0==err);
  err = cudaMalloc(&templateD, 2* templateSize*nImages*fpSize); assert(0==err);
  err = cudaMalloc(&  resultD, 2*   resultSize*nImages*fpSize); assert(0==err);

  /* copying data from the host to the device */


  const int TILE2 = 4;
  const int TILE  = 1<<TILE2;
  if (transpose)
  {
    {
      err = cudaMemcpy(imageD + imageSize*nImages, imageH, imageSize*nImages*fpSize, cudaMemcpyHostToDevice); assert(0==err);
      const int xTiles = ALIGN<TILE2>(imageNX) >> TILE2;
      const int yTiles = ALIGN<TILE2>(imageNY) >> TILE2;
      dev_transpose<TILE><<<dim3(xTiles*yTiles, nImages), dim3(TILE, TILE)>>>(
          imageD, imageD + imageSize*nImages, imageNX, imageNY, xTiles);
    }

    {
      err = cudaMemcpy(templateD + templateSize*nImages, templateH, templateSize*nImages*fpSize, cudaMemcpyHostToDevice); assert(0==err);
      const int xTiles = ALIGN<TILE2>(templateNX) >> TILE2;
      const int yTiles = ALIGN<TILE2>(templateNY) >> TILE2;
      dev_transpose<TILE><<<dim3(xTiles*yTiles, nImages), dim3(TILE, TILE)>>>(
          templateD, templateD + templateSize*nImages, templateNX, templateNY, xTiles);
    }

    std::swap(imageNX,    imageNY);
    std::swap(templateNX, templateNY);
    std::swap(resultNX,   resultNY);
  }
  else
  {
    err = cudaMemcpy(   imageD,    imageH,    imageSize*nImages*fpSize, cudaMemcpyHostToDevice); assert(0==err);
    err = cudaMemcpy(templateD, templateH, templateSize*nImages*fpSize, cudaMemcpyHostToDevice); assert(0==err);
  }


  /* copy constants */

  err = cudaMemcpyToSymbol("nImages",   &nImages,   sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);

  err = cudaMemcpyToSymbol("imageNX",   &imageNX,   sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("imageNY",   &imageNY,   sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("imageSize", &imageSize, sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);

  err = cudaMemcpyToSymbol("templateNX",    &templateNX,    sizeof(int  ), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("templateNY",    &templateNY,    sizeof(int  ), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("templateSize",  &templateSize,  sizeof(int  ), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("templateCoeff", &templateCoeff, sizeof(float), 0, cudaMemcpyHostToDevice); assert(0==err);

  err = cudaMemcpyToSymbol("resultNX",   &resultNX,   sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("resultNY",   &resultNY,   sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);
  err = cudaMemcpyToSymbol("resultSize", &resultSize, sizeof(int), 0, cudaMemcpyHostToDevice); assert(0==err);

#ifdef _DEBUG_
  fprintf(stderr, " -- Computing normalized correlation matrix -- \n");
#endif

  /* subtract average value from the temlate */

#if 1
  {
    const dim3 grid(nImages, 1, 1);
    dev_computeTemplate<512><<<grid, 512>>>(templateD);
  }
#endif

  /* compute correlation matrix */
  /* here we can use cuFFT to compute correlation matrix 
   * of imageD and templateD 
   */

#if 1
  {
#ifdef _FERMI_
    const int NPT = 8;
#else
    const int NPT = 3;
#endif
    const dim3 grid(nImages, (resultNY-1)/NPT+1, 1);
    if      (imageNX <=   64) dev_nxcor<  64,NPT><<<grid,  64>>>(imageD, templateD, resultD);
    else if (imageNX <=  128) dev_nxcor< 128,NPT><<<grid, 128>>>(imageD, templateD, resultD);
    else if (imageNX <=  192) dev_nxcor< 192,NPT><<<grid, 192>>>(imageD, templateD, resultD);
    else if (imageNX <=  256) dev_nxcor< 256,NPT><<<grid, 256>>>(imageD, templateD, resultD);
    else if (imageNX <=  384) dev_nxcor< 384,NPT><<<grid, 384>>>(imageD, templateD, resultD);
    else if (imageNX <=  512) dev_nxcor< 512,NPT><<<grid, 512>>>(imageD, templateD, resultD);
#ifdef _FERMI_  /* GT200 cannot run more than 512 threads/block */
    else if (imageNX <=  640) dev_nxcor< 640,NPT><<<grid, 640>>>(imageD, templateD, resultD);
    else if (imageNX <=  768) dev_nxcor< 768,NPT><<<grid, 768>>>(imageD, templateD, resultD);
    else if (imageNX <=  896) dev_nxcor< 896,NPT><<<grid, 896>>>(imageD, templateD, resultD);
    else if (imageNX <= 1024) dev_nxcor<1024,NPT><<<grid,1024>>>(imageD, templateD, resultD);
#endif
    else assert(0);
  }
#endif

  /* normalize correlation matrix */

#if 1
  {
    const dim3 grid(nImages, 1, 1);
    if      (imageNX <=   64) dev_normalizeCorr< 6><<<grid,  64>>>(imageD, templateD, resultD);
    else if (imageNX <=  128) dev_normalizeCorr< 7><<<grid, 128>>>(imageD, templateD, resultD);
    else if (imageNX <=  256) dev_normalizeCorr< 8><<<grid, 256>>>(imageD, templateD, resultD);
    else if (imageNX <=  512) dev_normalizeCorr< 9><<<grid, 512>>>(imageD, templateD, resultD);
#ifdef _FERMI_
    else if (imageNX <= 1024) dev_normalizeCorr<10><<<grid,1024>>>(imageD, templateD, resultD);
#endif
    else assert(0);
  }
#endif


  /* copying result from the device to the host */

  if (transpose)
  {
    const int xTiles = ALIGN<TILE2>(resultNX) >> TILE2;
    const int yTiles = ALIGN<TILE2>(resultNY) >> TILE2;
    dev_transpose<TILE><<<dim3(xTiles*yTiles, nImages), dim3(TILE, TILE)>>>(
        resultD + resultSize*nImages, resultD, resultNX, resultNY, xTiles);
    err = cudaMemcpy(resultH, resultD + resultSize*nImages, resultSize*nImages*fpSize,  cudaMemcpyDeviceToHost); assert(0==err);
  }
  else
    err = cudaMemcpy(resultH, resultD, resultSize*nImages*fpSize,  cudaMemcpyDeviceToHost); assert(0==err);

#if 0
  for (int i = 0 ; i < 16; i++)
    fprintf(stderr, "i= %d resultH= %g\n", i,resultH[i + resultNX*3]);
#endif

  /* freeing DRAM */

#ifdef _DEBUG_
  fprintf(stderr, " -- Freeing  DRAM -- \n");
#endif

  err = cudaFree(   imageD); assert(0==err);
  err = cudaFree(templateD); assert(0==err);
  err = cudaFree(  resultD); assert(0==err);
}

  template<typename T>
void transpose(T *dst, const T *src, const int nx, const int ny)
{
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      dst[i*ny+j] = src[j*nx + i];
}


#ifdef __cplusplus
extern "C"
#endif
void cunxcor(
    const float *   imageH,
    const float *templateH,
    __out float *  resultH,
    const int      nImages,
    const int      imageNY,
    const int      imageNX,
    const int   templateNY,
    const int   templateNX)
{
  const int nImagesPerBatch = 10000;

  const int resultNX = imageNX - templateNX + 1;
  const int resultNY = imageNY - templateNY + 1;
  assert(resultNX > 0);
  assert(resultNY > 0);


  const int    imageSize =    imageNX *    imageNY;
  const int templateSize = templateNX * templateNY;
  const int   resultSize =   resultNX *   resultNY;

  for (int i = 0; i < nImages; i += nImagesPerBatch)
    cunxcor_main(
        imageH    + i*   imageSize,
        templateH + i*templateSize,
        resultH   + i*  resultSize,
        nImages   - i > nImagesPerBatch ? nImagesPerBatch : nImages - i,
        imageNY, imageNX, templateNY, templateNX,
        imageNX < imageNY);
}


