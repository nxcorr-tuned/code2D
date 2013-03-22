#if 1
#define CUDA
#endif

#include <stdio.h>
#include <math.h>
#include "nxcor.h"
#ifdef CUDA
#include "cunxcor.h"
#endif
#include<stdlib.h>
#include <assert.h>
#include "clocks.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct 
{
  int nx, ny;
} mat;

#define NIMG 5
const mat img2D[NIMG] = {{512,19},{256,19},{128,19},{64,19},{32,19}};
int main()
{
  int i;
  for (i= 0; i < NIMG; i++)
  {
    const int nm = 64*64;
    const int mi = img2D[i].nx;
    const int ni = img2D[i].ny;
    fprintf(stderr, "img= %d:  %d x %d \n", i, mi, ni);
    int ntmpl = 0;
    const int nt = 9;
    int mt = mi;
    while (mt > 16)
    {
      mt *= 0.5;
      fprintf(stderr, "  tmpl= %d:  %d x %d \n", ntmpl++, mt, nt);

      const int mr = mi-mt+1;
      const int nr = ni-nt+1;

      float *image = malloc(sizeof(float)*mi*ni*nm);
      float *template = malloc(sizeof(float)*mt*nt*nm);

      float *result = malloc(sizeof(float)*mr*nr*nm);
      float *curesult = malloc(sizeof(float)*mr*nr*nm);

      int i;
      for (i=0; i<mi*ni*nm; i++)
        image[i] = (float)rand()/RAND_MAX*100.0;
      for (i=0; i<mt*nt*nm; i++)
        template[i] = (float)rand()/RAND_MAX*100.0;

      double t0,t1,t2, t3,t4,t5;
      fprintf(stderr, "starting cpu computation\n");
      t0 = wallclock();
      nxcor(image, template, result, 
          nm,mi,ni,mt,nt);
      t1 = wallclock() - t0;
      fprintf(stderr, "end of cpu computation\n");
      //  fprintf(stderr, "t1: %f\n",t1);
      //  return 0;
      for (i=0; i<nm*mr*nr; i++)
        curesult[i]=0;
      float som;
#ifdef CUDA
      fprintf(stderr, "starting gpu computation\n");
      t0 = wallclock();
      cunxcor(image, template, curesult, 
          nm,mi,ni,mt,nt);
      t2 = wallclock() - t0;
      fprintf(stderr, "end of gpu computation\n");
      fprintf(stderr, "starting gpu computation\n");
      t0 = wallclock();
      cunxcor(image, template, curesult, 
          nm,mi,ni,mt,nt);
      t3 = wallclock() - t0;
      fprintf(stderr, "end of gpu computation\n");
      som=0;
      for (i=0; i<nm*mr*nr; i++)
        som += fabsf(result[i]-curesult[i]);
      fprintf(stderr, ">>>>>>>>>>>>>>>som: %g\n",som/(nm*mr*nr));
#endif
      fprintf(stderr, "starting SSE computation\n");
      t0 = wallclock();
      nxcor_sse(image, template, curesult, 
          nm,mi,ni,mt,nt);
      t4 = wallclock() - t0;
      fprintf(stderr, "end of SSE computation\n");
      //  for (i=0; i<100; i++)
      //    fprintf(stderr, "%f %f\n",result[i],curesult[i]);
      som=0;
      for (i=0; i<nm*mr*nr; i++)
        som += fabsf(result[i]-curesult[i]);
      fprintf(stderr, ">>>>>>>>>>>>>>>som: %g\n",som/(nm*mr*nr));
#ifdef __AVX__
      fprintf(stderr, "starting AVX computation\n");
      t0 = wallclock();
      nxcor_avx(image, template, curesult, 
          nm,mi,ni,mt,nt);
      t5 = wallclock() - t0;
      fprintf(stderr, "end of AVX computation\n");
      //  for (i=0; i<100; i++)
      //    fprintf(stderr, "%f %f\n",result[i],curesult[i]);
      som=0;
      for (i=0; i<nm*mr*nr; i++)
        som += fabsf(result[i]-curesult[i]);
      fprintf(stderr, ">>>>>>>>>>>>>>>som: %g\n",som/(nm*mr*nr));
#endif
      fprintf(stderr, "t1: %g\n",t1);
      fprintf(stderr, "t2: %g\n",t2);
      fprintf(stderr, "t3: %g  bandwidth= %g GB/s  %g GFLOP/s\n",t3, 
          (double)mr*nr*mt*nt*nm*2*4/t3/1.0e9,
          (double)mr*nr*mt*nt*nm*2/t3/1.0e9);
      fprintf(stderr, "t4: %g  bandwidth= %g GB/s  %g GFLOP/s\n",t4,
          (double)mr*nr*mt*nt*nm*2*4/t4/1.0e9,
          (double)mr*nr*mt*nt*nm*2/t4/1.0e9);
#ifdef __AVX__
      fprintf(stderr, "t5: %g  bandwidth= %g GB/s  %g GFLOP/s\n",t4,
          (double)mr*nr*mt*nt*nm*2*4/t5/1.0e9,
          (double)mr*nr*mt*nt*nm*2/t5/1.0e9);
#endif
      fprintf(stderr, "TD: all is well that ends well\n");
    }
  }
  return 0;
}
