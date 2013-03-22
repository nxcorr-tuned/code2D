#ifdef __AVX__
#pragma once

#include "avx.h"
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cfloat>

#define __out

template<const int SIZE2>
inline size_t ALIGN(const size_t n) {return ((n-1)&(-(1<<SIZE2))) + (1<<SIZE2);}

template<const int SIZE2, typename T>
inline T* ALIGN(T* ptr)
{
  const size_t SIZE = 1 << SIZE2;
  char*  ptr_u = (char*)ptr;
  return (T*)((size_t)ptr_u + (SIZE - (size_t)ptr_u % SIZE));
}

#define CHK_ALIGN(ptr, align) assert(0 == (size_t)(ptr) % align)

#ifdef __cplusplus
extern "C"
#endif
void nxcor_avx(
    const float *   imageH,
    const float *templateH,
    __out float *  resultH,
    const int      nImages,
    const int      imageNY,
    const int      imageNX,
    const int   templateNY,
    const int   templateNX)
{
  assert(   imageNX > 0);
  assert(   imageNY > 0);
  assert(templateNX > 0);
  assert(templateNY > 0);

  const int     resultNX = imageNX - templateNX + 1;
  const int     resultNY = imageNY - templateNY + 1;
  assert(resultNX > 0);
  assert(resultNY > 0);

  const int    imageSize =    imageNX *    imageNY;
  const int templateSize = templateNX * templateNY;
  const int   resultSize =   resultNX *   resultNY;

  const float templateCoeff = 1.0f/(float)templateSize;

  const int    imageNXa = ALIGN<3>(   imageNX);
  const int templateNXa = ALIGN<3>(templateNX);

  const int    imageSize_a =    imageNXa *    imageNY;
  const int templateSize_a = templateNXa * templateNY;

#pragma omp parallel
  {
    std::vector<float> templateU(8*templateSize_a+32, 0.0f);
    float *templateA = ALIGN<5>(&templateU[0]);
    CHK_ALIGN(templateA, 32);

    std::vector<float> image1u(8*imageSize_a+32);
    std::vector<float> image2u(8*imageSize_a+32);
    std::vector<float> image3u(8*imageSize_a+32);
    std::vector<float> image4u(8*imageSize_a+32);
    std::vector<float> image5u(8*imageSize_a+32);
    std::vector<float> image6u(8*imageSize_a+32);
    std::vector<float> image7u(8*imageSize_a+32);
    std::vector<float> image8u(8*imageSize_a+32);

    float *image1 = ALIGN<5>(&image1u[0]);
    float *image2 = ALIGN<5>(&image2u[0]);
    float *image3 = ALIGN<5>(&image3u[0]);
    float *image4 = ALIGN<5>(&image4u[0]);
    float *image5 = ALIGN<5>(&image5u[0]);
    float *image6 = ALIGN<5>(&image6u[0]);
    float *image7 = ALIGN<5>(&image7u[0]);
    float *image8 = ALIGN<5>(&image8u[0]);

    CHK_ALIGN(image1, 32);
    CHK_ALIGN(image2, 32);
    CHK_ALIGN(image3, 32);
    CHK_ALIGN(image4, 32);
    CHK_ALIGN(image5, 32);
    CHK_ALIGN(image6, 32);
    CHK_ALIGN(image7, 32);
    CHK_ALIGN(image8, 32);
  


#pragma omp for
    for (int imageIdx = 0; imageIdx < nImages; imageIdx++)
    {
      /* compute pointers to an image about to be processed */

      const float *   imageX =    imageH + imageIdx *    imageSize;
      const float *templateX = templateH + imageIdx * templateSize;
      __out float *  resultX =   resultH + imageIdx *   resultSize;


      /* copy image and tempalte with proper padding */

      for (int j = 0; j < imageNY; j++)
          for (int i = 0; i < imageNX; i++)
          {
                       image1[j*imageNXa + i - 0] = imageX[j*imageNX + i];
            if (i > 0) image2[j*imageNXa + i - 1] = imageX[j*imageNX + i];
            if (i > 1) image3[j*imageNXa + i - 2] = imageX[j*imageNX + i];
            if (i > 2) image4[j*imageNXa + i - 3] = imageX[j*imageNX + i];
            if (i > 3) image5[j*imageNXa + i - 4] = imageX[j*imageNX + i];
            if (i > 4) image6[j*imageNXa + i - 5] = imageX[j*imageNX + i];
            if (i > 5) image7[j*imageNXa + i - 6] = imageX[j*imageNX + i];
            if (i > 6) image8[j*imageNXa + i - 7] = imageX[j*imageNX + i];
          }

      for (int j = 0; j < templateNY; j++)
        for (int i = 0; i < templateNX; i++)
          templateA[j*templateNXa + i] = templateX[j*templateNX + i];

      /**********************/
      /**********************/
      /**********************/

      /* remove mean value from the template */

#if 1
      {
        double templateSum = 0.0;
        for (int i = 0; i < templateSize; i++)
          templateSum += templateX[i];

        templateSum *= templateCoeff;

        for (int j = 0; j < templateSize_a; j += templateNXa)
          for (int i = 0; i < templateNX; i++)
            templateA[j+i] -= templateSum;
      }
#endif


      /**********************/
      /**********************/
      /**********************/

      /* compute correlation matrix */

#if 1  /* hot-spot, consumes >95% of the runtime, SSE is essential */
      {
        _v8sf *templateV = (_v8sf*)&templateA[0];

        for (int j = 0; j < resultNY; j += 4)
          for (int i = 0; i < resultNX; i++)
          {
            _v8sf *imageV = (_v8sf*)&image1[i&(-8)];
            switch(i&7)
            {
              case 1:
                imageV = (_v8sf*)&image2[i&(-8)];
                break;
              case 2:
                imageV = (_v8sf*)&image3[i&(-8)];
                break;
              case 3:
                imageV = (_v8sf*)&image4[i&(-8)];
                break;
              case 4:
                imageV = (_v8sf*)&image5[i&(-8)];
                break;
              case 5:
                imageV = (_v8sf*)&image6[i&(-8)];
                break;
              case 6:
                imageV = (_v8sf*)&image7[i&(-8)];
                break;
              case 7:
                imageV = (_v8sf*)&image8[i&(-8)];
                break;
            }

            _v8sf corrCoeff1 = v8sf(0.0f);
            _v8sf corrCoeff2 = v8sf(0.0f);
            _v8sf corrCoeff3 = v8sf(0.0f);
            _v8sf corrCoeff4 = v8sf(0.0f);

            int taddr1 = 0;
            int taddr2 = taddr1 + (templateNXa>>3);
            int taddr3 = taddr2 + (templateNXa>>3);
            int taddr4 = taddr3 + (templateNXa>>3);

            int iaddr1 =      j * (imageNXa>>3);
            int iaddr2 = iaddr1 + (imageNXa>>3);
            int iaddr3 = iaddr2 + (imageNXa>>3);
            int iaddr4 = iaddr3 + (imageNXa>>3);
            int iaddr5 = iaddr4 + (imageNXa>>3);
            int iaddr6 = iaddr5 + (imageNXa>>3);
            int iaddr7 = iaddr6 + (imageNXa>>3);

            for (int row = 0; row < templateNY; row += 4)
            {

              /* prefetch data for the next pass */
              __builtin_prefetch(imageV+iaddr7+(imageNXa>>2));

              /* compute multiple correlation coefficients inside the inner loop */
              for (int col = 0; col < templateNXa>>3; col++)
              {
                const _v8sf tmp1 = templateV[taddr1+col];
                const _v8sf tmp2 = templateV[taddr2+col];
                const _v8sf tmp3 = templateV[taddr3+col];
                const _v8sf tmp4 = templateV[taddr4+col];

                corrCoeff1 += 
                    tmp1*imageV[iaddr1+col] 
                  + tmp2*imageV[iaddr2+col]
                  + tmp3*imageV[iaddr3+col] 
                  + tmp4*imageV[iaddr4+col];

                corrCoeff2 += 
                    tmp1*imageV[iaddr2+col] 
                  + tmp2*imageV[iaddr3+col]
                  + tmp3*imageV[iaddr4+col] 
                  + tmp4*imageV[iaddr5+col];

                corrCoeff3 += 
                    tmp1*imageV[iaddr3+col] 
                  + tmp2*imageV[iaddr4+col]
                  + tmp3*imageV[iaddr5+col] 
                  + tmp4*imageV[iaddr6+col];

                corrCoeff4 += 
                    tmp1*imageV[iaddr4+col] 
                  + tmp2*imageV[iaddr5+col]
                  + tmp3*imageV[iaddr6+col] 
                  + tmp4*imageV[iaddr7+col];
              }
              taddr1 += templateNXa>>1;
              taddr2 += templateNXa>>1;
              taddr3 += templateNXa>>1;
              taddr4 += templateNXa>>1;
              iaddr1 +=    imageNXa>>1;
              iaddr2 +=    imageNXa>>1;
              iaddr3 +=    imageNXa>>1;
              iaddr4 +=    imageNXa>>1;
              iaddr5 +=    imageNXa>>1;
              iaddr6 +=    imageNXa>>1;
              iaddr7 +=    imageNXa>>1;
            }
                                resultX[(j+0)*resultNX + i] = V4(__reduce_v8sf(corrCoeff1))[0];
            if (j+1 < resultNY) resultX[(j+1)*resultNX + i] = V4(__reduce_v8sf(corrCoeff2))[0];
            if (j+2 < resultNY) resultX[(j+2)*resultNX + i] = V4(__reduce_v8sf(corrCoeff3))[0];
            if (j+3 < resultNY) resultX[(j+3)*resultNX + i] = V4(__reduce_v8sf(corrCoeff4))[0];
          }
      }
#endif

      /**********************/
      /**********************/
      /**********************/

      /* normalize correlation matrix */

#if 1  /* <5% of runtime is spent in it, not sure if SSE version is needed */
      {
        double templateSum2 = 0.0;
        for (int i = 0; i < templateSize_a; i++)
          templateSum2 += templateA[i]*templateA[i];

        int iaddr = 0;
        const int windowSize = templateNY * imageNXa;

        std::vector< std::pair<double, double> > imageSums(resultNX, std::make_pair(0.0, 0.0));

        while (iaddr < windowSize)
        {
          image2[iaddr] = image1[iaddr]*image1[iaddr];
          for (int i = iaddr + 1; i < iaddr + imageNXa; i++)
          {
            image2[i] = image2[i-1] + image1[i]*image1[i];
            image1[i] = image1[i-1] + image1[i];
          }
          imageSums[0].first  += image1[iaddr + templateNX - 1];
          imageSums[0].second += image2[iaddr + templateNX - 1];
          for (int i = 1; i < resultNX; i++)
          {
            imageSums[i].first  += image1[iaddr+i-1 + templateNX] - image1[iaddr+i-1];
            imageSums[i].second += image2[iaddr+i-1 + templateNX] - image2[iaddr+i-1];
          }
          iaddr += imageNXa;
        }
        for (int i = 0; i < resultNX; i++)
        {
          const double imageSum  = imageSums[i].first;
          const double imageSum2 = imageSums[i].second;
          const float norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
          resultX[i] *= 1.0f/std::sqrt(norm2 + FLT_EPSILON); 
        }

        while (iaddr < imageSize_a)
        {
          image2[iaddr] = image1[iaddr]*image1[iaddr];
          for (int i = iaddr + 1; i < iaddr + imageNXa; i++)
          {
            image2[i] = image2[i-1] + image1[i]*image1[i];
            image1[i] = image1[i-1] + image1[i];
          }
          imageSums[0].first  -= image1[iaddr + templateNX - 1 - windowSize];
          imageSums[0].second -= image2[iaddr + templateNX - 1 - windowSize];
          imageSums[0].first  += image1[iaddr + templateNX - 1             ];
          imageSums[0].second += image2[iaddr + templateNX - 1             ];
          for (int i = 1; i < resultNX; i++)
          {
            imageSums[i].first  -= image1[iaddr+i-1 + templateNX - windowSize] - image1[iaddr + i-1 - windowSize];
            imageSums[i].second -= image2[iaddr+i-1 + templateNX - windowSize] - image2[iaddr + i-1 - windowSize];
            imageSums[i].first  += image1[iaddr+i-1 + templateNX             ] - image1[iaddr + i-1             ];
            imageSums[i].second += image2[iaddr+i-1 + templateNX             ] - image2[iaddr + i-1             ];
          }
          iaddr += imageNXa;

          const int raddr = (iaddr/imageNXa - templateNY) * resultNX;
          for (int i = 0; i < resultNX; i++)
          {
            const double imageSum  = imageSums[i].first;
            const double imageSum2 = imageSums[i].second;
            const float norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
            resultX[raddr + i] *= 1.0f/std::sqrt(norm2 + FLT_EPSILON); 
          }
        }

      }
#endif

    }
  }

#if 0
  for (int i = 0 ; i < 16; i++)
    fprintf(stderr, "i= %d resultH= %g\n", i,resultH[i + resultNX*1]);
  assert(0);
#endif


}
#endif /* __AVX__ */
