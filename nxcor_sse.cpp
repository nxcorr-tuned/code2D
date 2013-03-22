#include "sse.h"
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cfloat>

#define __out

template<const int SIZE2>
inline size_t ALIGN(const size_t n) {return ((n-1)&(-(1<<SIZE2))) + (1<<SIZE2);}

#ifdef __cplusplus
extern "C"
#endif
void nxcor_sse(
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

  const int    imageNXa = ALIGN<2>(   imageNX);
  const int templateNXa = ALIGN<3>(templateNX);

  const int    imageSize_a =    imageNXa *    imageNY;
  const int templateSize_a = templateNXa * templateNY;
  assert((imageNXa    & 3) == 0);
  assert((imageSize_a & 3) == 0);


#pragma omp parallel
  {
    std::vector<float>    imageA(   8*imageSize_a+32);
    std::vector<float>    imageB(   8*imageSize_a+32);
    std::vector<float>    imageC(   8*imageSize_a+32);
    std::vector<float>    imageD(   8*imageSize_a+32);
    std::vector<float> templateA(8*templateSize_a+32, 0.0f);

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
          imageA[j*imageNXa + i  ] = imageX[j*imageNX + i];
          if (i > 0) imageB[j*imageNXa + i-1] = imageX[j*imageNX + i];
          if (i > 1) imageC[j*imageNXa + i-2] = imageX[j*imageNX + i];
          if (i > 2) imageD[j*imageNXa + i-3] = imageX[j*imageNX + i];
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
        _v4sf *templateV = (_v4sf*)&templateA[0];

        for (int j = 0; j < resultNY; j += 4)
          for (int i = 0; i < resultNX; i++)
          {
            _v4sf *imageV = (_v4sf*)&imageA[i&(-4)];
            switch(i&3)
            {
              case 1:
                imageV = (_v4sf*)&imageB[i&(-4)];
                break;
              case 2:
                imageV = (_v4sf*)&imageC[i&(-4)];
                break;
              case 3:
                imageV = (_v4sf*)&imageD[i&(-4)];
                break;
            }

            _v4sf corrCoeff1 = v4sf(0.0f);
            _v4sf corrCoeff2 = v4sf(0.0f);
            _v4sf corrCoeff3 = v4sf(0.0f);
            _v4sf corrCoeff4 = v4sf(0.0f);

            int taddr1 =                     0 ;
            int taddr2 =        templateNXa>>2 ;
            int iaddr1 =      j * (imageNXa>>2);
            int iaddr2 = iaddr1 + (imageNXa>>2);
            int iaddr3 = iaddr2 + (imageNXa>>2);
            int iaddr4 = iaddr3 + (imageNXa>>2);
            int iaddr5 = iaddr4 + (imageNXa>>2);

            for (int row = 0; row < templateNY; row += 2)
            {
              
              /* prefetch data for the next pass */
              __builtin_prefetch(imageV+iaddr5+(imageNXa>>1));

              /* compute multiple correlation coefficients inside the inner loop */
              for (int col = 0; col < templateNXa>>2; col += 2)
              {
                const _v4sf tmp1 = templateV[taddr1+col  ];
                const _v4sf tmp2 = templateV[taddr1+col+1];
                const _v4sf tmp3 = templateV[taddr2+col  ];
                const _v4sf tmp4 = templateV[taddr2+col+1];

                corrCoeff1 += 
                    tmp1*imageV[iaddr1+col  ] 
                  + tmp2*imageV[iaddr1+col+1]
                  + tmp3*imageV[iaddr2+col  ] 
                  + tmp4*imageV[iaddr2+col+1];

                corrCoeff2 += 
                    tmp1*imageV[iaddr2+col  ] 
                  + tmp2*imageV[iaddr2+col+1]
                  + tmp3*imageV[iaddr3+col  ] 
                  + tmp4*imageV[iaddr3+col+1];

                corrCoeff3 += 
                    tmp1*imageV[iaddr3+col  ] 
                  + tmp2*imageV[iaddr3+col+1]
                  + tmp3*imageV[iaddr4+col  ] 
                  + tmp4*imageV[iaddr4+col+1];

                corrCoeff4 += 
                    tmp1*imageV[iaddr4+col  ] 
                  + tmp2*imageV[iaddr4+col+1]
                  + tmp3*imageV[iaddr5+col  ] 
                  + tmp4*imageV[iaddr5+col+1];
              }
              taddr1 += templateNXa>>1;
              taddr2 += templateNXa>>1;
              iaddr1 +=    imageNXa>>1;
              iaddr2 +=    imageNXa>>1;
              iaddr3 +=    imageNXa>>1;
              iaddr4 +=    imageNXa>>1;
              iaddr5 +=    imageNXa>>1;
            }
                                resultX[(j+0)*resultNX + i] = V4(__reduce_v4sf(corrCoeff1))[0];
            if (j+1 < resultNY) resultX[(j+1)*resultNX + i] = V4(__reduce_v4sf(corrCoeff2))[0];
            if (j+2 < resultNY) resultX[(j+2)*resultNX + i] = V4(__reduce_v4sf(corrCoeff3))[0];
            if (j+3 < resultNY) resultX[(j+3)*resultNX + i] = V4(__reduce_v4sf(corrCoeff4))[0];
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
          imageB[iaddr] = imageA[iaddr]*imageA[iaddr];
          for (int i = iaddr + 1; i < iaddr + imageNXa; i++)
          {
            imageB[i] = imageB[i-1] + imageA[i]*imageA[i];
            imageA[i] = imageA[i-1] + imageA[i];
          }
          imageSums[0].first  += imageA[iaddr + templateNX - 1];
          imageSums[0].second += imageB[iaddr + templateNX - 1];
          for (int i = 1; i < resultNX; i++)
          {
            imageSums[i].first  += imageA[iaddr+i-1 + templateNX] - imageA[iaddr+i-1];
            imageSums[i].second += imageB[iaddr+i-1 + templateNX] - imageB[iaddr+i-1];
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
          imageB[iaddr] = imageA[iaddr]*imageA[iaddr];
          for (int i = iaddr + 1; i < iaddr + imageNXa; i++)
          {
            imageB[i] = imageB[i-1] + imageA[i]*imageA[i];
            imageA[i] = imageA[i-1] + imageA[i];
          }
          imageSums[0].first  -= imageA[iaddr + templateNX - 1 - windowSize];
          imageSums[0].second -= imageB[iaddr + templateNX - 1 - windowSize];
          imageSums[0].first  += imageA[iaddr + templateNX - 1             ];
          imageSums[0].second += imageB[iaddr + templateNX - 1             ];
          for (int i = 1; i < resultNX; i++)
          {
            imageSums[i].first  -= imageA[iaddr+i-1 + templateNX - windowSize] - imageA[iaddr + i-1 - windowSize];
            imageSums[i].second -= imageB[iaddr+i-1 + templateNX - windowSize] - imageB[iaddr + i-1 - windowSize];
            imageSums[i].first  += imageA[iaddr+i-1 + templateNX             ] - imageA[iaddr + i-1             ];
            imageSums[i].second += imageB[iaddr+i-1 + templateNX             ] - imageB[iaddr + i-1             ];
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
