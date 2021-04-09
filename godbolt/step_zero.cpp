// -O3 -mfma -mavx -mavx2 -fopt-info-vec-all
#include <stdlib.h>


template <int n, typename scalar_t, typename index_t>
struct InterpolateAA {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i, int interp_size) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      scalar_t wts = *(scalar_t*)&data[1][i * strides[1]];
      scalar_t t = InterpolateAA<n - 1, scalar_t, index_t>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i, interp_size);
      scalar_t output = t * wts;
      for (int j=1; j<interp_size; j++) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = InterpolateAA<n - 1, scalar_t, index_t>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i, interp_size);
        output += t * wts;
      }
      return output;
  }
};

template <typename scalar_t, typename index_t>
struct InterpolateAA<1, scalar_t, index_t> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i, int interp_size) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      scalar_t wts = *(scalar_t*)&data[1][i * strides[1]];
      scalar_t t = *(scalar_t *)&src[ids];
      scalar_t output = t * wts;
      for (int j=1; j<interp_size; j++) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = *(scalar_t *)&src[ids];
        output += t * wts;
      }
      return output;
    }
};


template <int n, typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa(char* src, char** data, const int64_t* strides, int64_t i, int interp_size) {
  return InterpolateAA<n, scalar_t, index_t>::eval(src, data, strides, i, interp_size);
}


template <typename scalar_t, typename index_t, int out_ndims>
static inline void basic_loop_aa(char** data, const int64_t* strides, int64_t n, int interp_size) {
  char* dst = data[0];
  char* src = data[1];
  for (int64_t i = 0; i < n; i++) {
    *(scalar_t*)&dst[i * strides[0]] = interpolate_aa<out_ndims, scalar_t, index_t>(
        src + i * strides[1], &data[2], &strides[2], i, interp_size);
  }
}


static inline bool is_zero_stride(const int64_t* strides, int interp_size) {
  bool output = strides[0] == 0;
  for (int i=1; i<2 * interp_size; i++) {
    output &= (strides[i] == 0);
  }
  return output;
}

template <typename scalar_t, typename index_t>
static inline bool is_contiguous_stride(const int64_t* strides, int interp_size) {
  bool output = (strides[0] == sizeof(index_t)) && (strides[1] == sizeof(scalar_t));
  for (int i=2; i<2 * interp_size; i+=2) {
    output &= (strides[i] == sizeof(index_t)) && (strides[i + 1] == sizeof(scalar_t));
  }
  return output;
}

template <int N, int non_zero_stride_dim, typename scalar_t, typename index_t>
struct CheckAlmostAllZeroStrides {
  static inline bool eval(const int64_t* strides, int interp_size) {
    // N is dim index: N -> dim0, N-1 -> dim1, ...
    // non_zero_stride_dim should be out_dims - dim
    bool output;
    if (N == non_zero_stride_dim) {
      output = is_contiguous_stride<scalar_t, index_t>(strides, interp_size);
    } else {
      output = is_zero_stride(strides, interp_size);
    }
    return output &&
      CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, scalar_t, index_t>::eval(
        &strides[2 * interp_size], interp_size);
  }
};

template <int non_zero_stride_dim, typename scalar_t, typename index_t>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, scalar_t, index_t> {
  static inline bool eval(const int64_t* strides, int interp_size) {
    return true;
  }
};

template <int n, int s, typename scalar_t, typename index_t>
static inline bool check_almost_all_zero_stride(const int64_t* strides, int interp_size) {
  return CheckAlmostAllZeroStrides<n, s, scalar_t, index_t>::eval(strides, interp_size);
}


using index_t = int64_t;
using scalar_t = float;
constexpr int out_ndims = 2;


void func(char** data, const int64_t* strides, int64_t n, int interp_size)
{

    if (
      (strides[0] == sizeof(scalar_t)) && (strides[1] == 0)
      && check_almost_all_zero_stride<out_ndims, 1, scalar_t, index_t>(&strides[2], interp_size)
    )
    {      // contiguous channels-last case
      basic_loop_aa<scalar_t, index_t, out_ndims>(data, strides, n, interp_size);
    }
}
