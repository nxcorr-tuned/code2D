// This software is adapted from:
//
//                       Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//

#include <float.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nxcor.h"
#ifdef USEMKL
#include <mkl_cblas.h>
#endif

// #define MEMCHECK

int align(int x)
{
  int n=4;
  return ((x-1)/n+1)*n;
}

void icvcalculatebuffersizes(int mi,int ni,int mt,int nt,
    int *imgbufsize, int *templbufsize, int *sumbufsize,
    int * sqsumbufsize, int*resnumbufsize, int*resdenombufsize)
{
  *imgbufsize      = align(nt*mi+ni);
  *templbufsize    = align(nt*mt);
  *resnumbufsize   = align(mi-mt+1);

  *sumbufsize      = align(mi);
  *sqsumbufsize    = align(mi);
  *resdenombufsize = align((mi-mt+1)*2);
}

void icvmatchtemplategetbufsize_coeffnormed(int mi,int ni,int mt,int nt,size_t *buffersize)
{
  int imgbufsize,templbufsize,sumbufsize,
      sqsumbufsize,resnumbufsize,resdenombufsize;
  size_t depth = sizeof(float);
  icvcalculatebuffersizes(mi,ni,mt,nt,&imgbufsize,&templbufsize,&sumbufsize,
      &sqsumbufsize,&resnumbufsize,&resdenombufsize);
  *buffersize = depth*(imgbufsize+templbufsize)+sizeof(float)*(sumbufsize+
      sqsumbufsize+resnumbufsize+resdenombufsize);
}

void icvmatchtemplateentry(int mi,int ni,int mt,int nt,
    const float* image, const float* template, float*buffer,
    float **imgbuf, float ** templbuf,
    float **sumbuf, float **sqsumbuf, float **resnum, float **resdenom)
{
  int i;
  int templbufsize = 0, imgbufsize = 0, sumbufsize = 0, 
      sqsumbufsize = 0, resnumbufsize = 0, resdenombufsize = 0;

  icvcalculatebuffersizes(mi,ni,mt,nt,&imgbufsize,&templbufsize,
      &sumbufsize,&sqsumbufsize,&resnumbufsize,&resdenombufsize);
#ifdef MEMCHECK
  size_t depth = sizeof(float);
#ifdef _OPENMP
#pragma omp critical
#endif
  {
    *templbuf = malloc(templbufsize*depth);              assert(*templbuf !=0);
    *imgbuf   = malloc(imgbufsize*depth);                assert(*imgbuf   !=0);
    *resnum   = malloc(resnumbufsize*sizeof(float));     assert(*resnum   !=0);
    *sumbuf   = malloc(sumbufsize*sizeof(float));        assert(*sumbuf   !=0);
    *sqsumbuf = malloc(sqsumbufsize*sizeof(float));      assert(*sqsumbuf !=0);
    *resdenom = malloc(resdenombufsize*sizeof(float));   assert(*resdenom !=0);
  }
#else
  int p = 0;
  *templbuf = &buffer[p];
  p += templbufsize;
  *imgbuf   = &buffer[p];
  p += imgbufsize;
  *resnum   = &buffer[p];
  p += resnumbufsize;
  *sumbuf   = &buffer[p];
  p += sumbufsize;
  *sqsumbuf = &buffer[p];
  p += sqsumbufsize;
  *resdenom = &buffer[p];
  p += resdenombufsize;
#endif

  for (i=0; i< mi; i++)
  {
    int j;
    for (j=0; j<nt; j++)
      (*imgbuf)[i*nt+j] = image[i*ni + j];
  }
  for (i=0; i<mt; i++)
  {
    int j;
    for (j=0; j<nt; j++)
      (*templbuf)[i*nt+j] = template[i*nt + j];
  }
}

float icvcrosscorr(const float *vec1, const float *vec2, int len )
{
#ifdef USEMKL
  return cblas_sdot(len, vec1, 1, vec2, 1);
#else

  float sum = 0;
  int i;

  for( i = 0; i <= len - 4; i += 4 )
  {
    float v0 = vec1[i] * vec2[i];
    float v1 = vec1[i + 1] * vec2[i + 1];
    float v2 = vec1[i + 2] * vec2[i + 2];
    float v3 = vec1[i + 3] * vec2[i + 3];

    sum += v0 + v1 + v2 + v3;
  }
  for( ; i < len; i++ )
  {
    float v = vec1[i] * vec2[i];

    sum += v;
  }
  return sum;
#endif
}

float icvsumpixels(const float *vec, int len)
{
  float sum = 0;
  int i;

  for( i = 0; i <= len - 4; i += 4 )
  {
    sum += vec[i] + vec[i + 1] + vec[i + 2] + vec[i + 3];
  }

  for( ; i < len; i++ )
  {
    sum += vec[i];
  }
  return sum;
}

void icvmatchtemplate17(float*image, float*template, 
    float*result,int mi,int ni,int mt,int nt,void*buffer)
{
  float *imgbuf = 0;
  float *templbuf = 0;
  float *sumbuf = 0;
  float *sqsumbuf = 0;
  float *resnum = 0;
  float *resdenom = 0;
  float templcoeff = 0;
  float templsum = 0;

  int mr = mi-mt+1;
  int nr = ni-nt+1;
  int winlen = mt*nt;
  float wincoeff = 1.0f / (winlen + FLT_EPSILON);
  int x,y;

  icvmatchtemplateentry(mi,ni, mt, nt,
      image, template, buffer,
      &imgbuf, &templbuf,
      &sumbuf, &sqsumbuf, &resnum, &resdenom);

  {
    const float *rowptr = (const float *) imgbuf;
    float templsqsum = icvcrosscorr(template,template,winlen);
    templsum = icvsumpixels(template,winlen);
    templcoeff = templsqsum - (templsum * templsum * wincoeff);

    templcoeff = 1.0f/sqrtf(fabsf(templcoeff)+FLT_EPSILON);

    // height == m
    // width  == n
    for (y=0; y<mi; y++, rowptr += nt)
    {
      sumbuf[y]   = icvsumpixels(rowptr, nt);
      sqsumbuf[y] = icvcrosscorr(rowptr, rowptr, nt);
    }
  }

  for (x=0; x<nr; x++)
  {
    float sum=0.0f;
    float sqsum = 0.0f;
    float *imgptr = imgbuf + x;
    if (x > 0)
    {
      const float *src = image + x + nt -1;
      float *dst = imgptr - 1;
      float out_val = dst[0];

      dst += nt;

      for (y=0; y<mi; y++, src+=ni, dst += nt)
      {
        float in_val = src[0];
        sumbuf[y] += in_val - out_val;
        sqsumbuf[y] += (in_val - out_val) * (in_val + out_val);
        out_val = dst[0];
        dst[0] = in_val;
      }
    }
    for (y=0; y<mt; y++)
    {
      sum += sumbuf[y];
      sqsum += sqsumbuf[y];
    }
    for( y = 0; y < mr; y++, imgptr += nt )
    {
      float res = icvcrosscorr( imgptr, templbuf, winlen );

      if( y > 0 )
      {
        sum -= sumbuf[y - 1];
        sum += sumbuf[y + mt - 1];
        sqsum -= sqsumbuf[y - 1];
        sqsum += sqsumbuf[y + mt - 1];
      }
      resnum[y] = res;
      resdenom[y] = sum;
      resdenom[y + mr] = sqsum;
    }
    for( y = 0; y < mr; y++ )
    {
      float sum = resdenom[y];
      float wsum = wincoeff * sum;
      float res = resnum[y] - wsum * templsum;
      float nrm_s = resdenom[y + mr] - wsum * sum;

      res *= templcoeff / sqrtf( fabs( nrm_s ) + FLT_EPSILON );
      result[x + y * nr] = res;
    }
  }
#ifdef MEMCHECK
#ifdef _OPENMP
#pragma omp critical
#endif
  {
    free(imgbuf);
    free(templbuf);
    free(sumbuf);
    free(sqsumbuf);
    free(resnum);
    free(resdenom);
  }
#endif
}

void cvmatchtemplate(float* image,float* template,float*result,
    int mi,int ni,int mt,int nt, void* buffer)
{
  icvmatchtemplate17(image,template,result,mi,ni,mt,nt,(float*)buffer);
}
void nxcor(float*image, float*template, float*result, 
    int nm, int mi, int ni, int mt, int nt)
{
  size_t buffersize;
  icvmatchtemplategetbufsize_coeffnormed(mi,ni,mt,nt,&buffersize);
  int nthreads,mythread;
  char *buffer;
  int mr = mi - mt + 1;
  int nr = ni - nt + 1;
#pragma omp parallel default(none) \
  shared(image,template,result,ni,mi,nt,mt,nm,nthreads,\
      buffersize,buffer,mr,nr)\
  private(mythread)
  {
#ifdef _OPENMP
    mythread = omp_get_thread_num();
#else
    mythread = 0;
#endif
#pragma omp master
    {
#ifdef _OPENMP
      nthreads = omp_get_num_threads();
#else
      nthreads = 1;
#endif
      buffer = malloc(nthreads*buffersize);
      assert(buffer != 0);
    }
#pragma omp barrier
    {
      int mm;
#pragma omp for
      for (mm = 0; mm<nm; mm++)
      {
        size_t boff = (size_t)mythread*buffersize;
        size_t ioff = (size_t)mm*mi*ni;
        size_t toff = (size_t)mm*mt*nt;
        size_t roff = (size_t)mm*mr*nr;
        //	nxcor(&image[ioff], &template[toff], &result[roff], 
        //	      (int)ni, (int)mi, (int)nt, (int)mt,&buffer[boff]);
        icvmatchtemplate17(&image[ioff],&template[toff],&result[roff],
            (int)mi,(int)ni,(int)mt,(int)nt,&buffer[boff]);
      }
    }
  }
  free(buffer);
}
