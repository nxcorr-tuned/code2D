#pragma once
#include "sse.h"

typedef int    _v8si  __attribute__((vector_size(32)));
typedef float  _v8sf  __attribute__((vector_size(32)));
typedef double _v4df  __attribute__((vector_size(32)));

inline _v8sf v8sf(const float x) {return (_v8sf){x,x,x,x,x,x,x,x}; }
inline _v8si v8si(const  int  x) {return (_v8si){x,x,x,x,x,x,x,x}; }

union V8
{
  _v8sf x;
  float v[8];
  V8() {}
  V8(const _v8sf _x) : x(_x) {}
  float operator[](const int i) const {return v[i];}
  operator _v8sf() const {return x;}
};

inline _v8sf __bcast0(const _v4sf *x0)
{
  return __builtin_ia32_vbroadcastf128_ps256(x0);
}

inline _v8sf pack2ymm(const _v4sf x0, const _v4sf x1)
{ /* merges two xmm  into one ymm */
  _v8sf ymm = {};
  ymm = __builtin_ia32_vinsertf128_ps256(ymm, x0, 0);
  ymm = __builtin_ia32_vinsertf128_ps256(ymm, x1, 1);
  return ymm;
}
template<const int ch>
inline _v4sf __extract(const _v8sf x)
{  /* extracts xmm from ymm */
  assert(ch >= 0 && ch < 2);
  return __builtin_ia32_vextractf128_ps256(x, ch);
}
template<const bool hi1, const bool hi2>
inline _v8sf __merge(const _v8sf x, const _v8sf y)
{ /* merges two ymm into one ymm(x[k1],y[k2]), k = hi ? 0-127:128-255 */
  const int mask = 
    hi1 ? 
    (hi2 ? 0x31 : 0x21) :
    (hi2 ? 0x30 : 0x20);
  return __builtin_ia32_vperm2f128_ps256(x, y, mask);
}
inline _v8sf __mergelo(const _v8sf x, const _v8sf y)
{ /* merges [0-127] of two ymm into one ymm(x[0-127],y[0-127]) */
  return __builtin_ia32_vperm2f128_ps256(x, y, 0x20);
}
inline _v8sf __mergehi(const _v8sf x, const _v8sf y)
{ /* merges [128-255] of two ymm into one ymm(x[128-255],y[128-255]) */
  return __builtin_ia32_vperm2f128_ps256(x, y, 0x31);
}
#include <cassert>
template<const int N>
inline _v8sf __bcast(const _v8sf x)
{ /* broadcast a Nth channel of ymm into all channels */
  assert(N < 8);
  const int NN = N & 3;
  const int mask = 
    NN == 0 ? 0 :
    NN == 1 ? 0x55 :
    NN == 2 ? 0xAA  : 0xFF;
    
  const _v8sf tmp = __builtin_ia32_shufps256(x, x, mask);

  return N < 4 ? __mergelo(tmp, tmp) : __mergehi(tmp,tmp);
}
template<const int N>
inline _v8sf __bcast2(const _v8sf x)
{ /* broadcast n<4 challen of 0-127 & 128-255 into each channels of 0-127&128-255 respectively */
  assert(N < 4);
  const int mask = 
    N == 0 ? 0 :
    N == 1 ? 0x55 :
    N == 2 ? 0xAA  : 0xFF;
    
  return __builtin_ia32_shufps256(x, x, mask);
}

inline _v4sf __reduce_v8sf(const _v8sf v8)
{
  const _v4sf a = __extract<0>(v8);
  const _v4sf b = __extract<1>(v8);
  return __reduce_v4sf(a) + __reduce_v4sf(b);
}



