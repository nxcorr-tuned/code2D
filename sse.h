#pragma once

#include <cassert>

typedef int    _v4si  __attribute__((vector_size(16)));
typedef float  _v4sf  __attribute__((vector_size(16)));
typedef double _v2df  __attribute__((vector_size(16)));

inline _v4sf v4sf(const float x) {return (_v4sf){x,x,x,x};}
inline _v4si v4si(const  int  x) {return (_v4si){x,x,x,x};}

inline _v4sf __reduce_v4sf(const _v4sf v)
{
  _v4sf a = __builtin_ia32_haddps(v, v);
  a = __builtin_ia32_haddps(a,a);
  return __builtin_ia32_shufps(a, a, 0x00);
}

union V4
{
  _v4sf x;
  float v[4];
  V4() {}
  V4(const _v4sf _x) : x(_x) {}
  float operator[](const int i) const {return v[i];}
};

union V2
{
  _v2df x;
  double v[2];
  V2() {}
  V2(const _v2df _x) : x(_x) {}
  double operator[](const int i) const {return v[i];}
};

template<const int N>
inline _v4sf __rotl_v4sf(const _v4sf x)
{
  switch(N&3)
  {
    case 1:
      return __builtin_ia32_shufps(x, x, (1<<0) + (2<<2) + (3<<4) + (0<<6));
    case 2:
      return __builtin_ia32_shufps(x, x, (2<<0) + (3<<2) + (0<<4) + (1<<6));
    case 3:
      return __builtin_ia32_shufps(x, x, (3<<0) + (0<<2) + (1<<4) + (2<<6));
    default:
      return x;
  };
};

  template<const int N>
inline _v4sf __rotr_v4sf(const _v4sf x)
{
  switch(N&3)
  {
    case 1:
      return __builtin_ia32_shufps(x, x, (3<<0) + (0<<2) + (1<<4) + (2<<6));
    case 2:
      return __builtin_ia32_shufps(x, x, (2<<0) + (3<<2) + (0<<4) + (1<<6));
    case 3:
      return __builtin_ia32_shufps(x, x, (1<<0) + (2<<2) + (3<<4) + (0<<6));
    default:
      return x;
  };
};

template<const int N>
inline _v4sf __bcast_v4sf(const _v4sf x)
{
  switch(N&3)
  {
    case 1:
      return __builtin_ia32_shufps(x,x,0x55);
    case 2:
      return __builtin_ia32_shufps(x,x,0xAA);
    case 3:
      return __builtin_ia32_shufps(x,x,0xFF);
    default:
      return __builtin_ia32_shufps(x,x,0x00);
  };
}

template<const bool ch0, const bool ch1, const bool ch2, const bool ch3>
inline _v4sf __select_v4sf(const _v4sf x, const _v4sf y)
{
  return __builtin_ia32_blendps(x, y, (ch0<<0) + (ch1<<1) + (ch2<<2) + (ch3<<3));
}

inline _v4sf __prefix_sum_v4sf(_v4sf x)
{
  x += __select_v4sf<0,1,1,1>(v4sf(0.0f), __rotr_v4sf<1>(x));
  x += __select_v4sf<0,0,1,1>(v4sf(0.0f), __rotr_v4sf<2>(x));
  return x;
}


