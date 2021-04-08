#include <torch/extension.h>

#include <cmath>
#include <vector>
#include <ATen/TypeDefault.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/UpSample.h>


#define USE_ALWAYS_INDEX64
// #define VERBOSE


namespace at {
namespace native {
namespace ti_upsample {

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;
using scale_t = std::vector<c10::optional<double>>;

#ifdef VERBOSE
static int TI_BASIC_LOOP_CHANNELS_FIRST_TRIGGERED = 0;
static int TI_BASIC_LOOP_CHANNELS_LAST_TRIGGERED = 0;
static int TI_BASIC_LOOP_FALLBACK_TRIGGERED = 0;
static bool TI_SHOW_STRIDES = true;
#endif

// Copied from aten/src/ATen/native/cpu/UpSampleMoreKernel.cpp
// for reuse if goes as a PR in PyTorch
template<typename scalar_t, typename index_t>
static inline void compute_source_index_and_lambda(
    index_t& input_index0,
    index_t& input_index1,
    scalar_t& lambda0,
    scalar_t& lambda1,
    scalar_t ratio,
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    input_index0 = static_cast<index_t>(output_index);
    input_index1 = static_cast<index_t>(output_index);
    lambda0 = static_cast<scalar_t>(1);
    lambda1 = static_cast<scalar_t>(0);
  } else {
    const scalar_t real_input_index = area_pixel_compute_source_index<scalar_t>(
        ratio, output_index, align_corners, /*cubic=*/false);
    input_index0 = static_cast<index_t>(real_input_index);
    index_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;
    lambda1 = real_input_index - static_cast<scalar_t>(input_index0);
    lambda0 = static_cast<scalar_t>(1.) - lambda1;
  }
}
// End of copied ...


template <int n, typename scalar_t, typename index_t, int interp_size>
struct Interpolate {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      scalar_t wts = *(scalar_t*)&data[1][i * strides[1]];
      scalar_t t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
      scalar_t output = t * wts;
      for (int j=1; j<interp_size; j++) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
        output += t * wts;
      }
      return output;
  }
};

template <typename scalar_t, typename index_t, int interp_size>
struct Interpolate<1, scalar_t, index_t, interp_size> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
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
struct Interpolate<n, scalar_t, index_t, 1> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      return Interpolate<n - 1, scalar_t, index_t, 1>::eval(src + ids, &data[2], &strides[2], i);
  }
};

template <typename scalar_t, typename index_t>
struct Interpolate<1, scalar_t, index_t, 1> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      return *(scalar_t *)&src[ids];
    }
};

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


template <int n, typename scalar_t, typename index_t, int interp_size>
static inline scalar_t interpolate(char* src, char** data, const int64_t* strides, int64_t i) {
  return Interpolate<n, scalar_t, index_t, interp_size>::eval(src, data, strides, i);
}

template <int n, typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa(char* src, char** data, const int64_t* strides, int64_t i, int interp_size) {
  return InterpolateAA<n, scalar_t, index_t>::eval(src, data, strides, i, interp_size);
}

template<int interp_size>
static inline bool is_zero_stride(const int64_t* strides) {
  bool output = strides[0] == 0;
  for (int i=1; i<2 * interp_size; i++) {
    output &= (strides[i] == 0);
  }
  return output;
}

template <typename scalar_t, typename index_t, int interp_size>
static inline bool is_contiguous_stride(const int64_t* strides) {
  bool output = (strides[0] == sizeof(index_t)) && (strides[1] == sizeof(scalar_t));
  for (int i=2; i<2 * interp_size; i+=2) {
    output &= (strides[i] == sizeof(index_t)) && (strides[i + 1] == sizeof(scalar_t));
  }
  return output;
}

template <int N, int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides {
  static inline bool eval(const int64_t* strides) {
    // N is dim index: N -> dim0, N-1 -> dim1, ...
    // non_zero_stride_dim should be out_dims - dim
    bool output;
    if (N == non_zero_stride_dim) {
      output = is_contiguous_stride<scalar_t, index_t, interp_size>(strides);
    } else {
      output = is_zero_stride<interp_size>(strides);
    }
    return output &&
      CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, scalar_t, index_t, interp_size>::eval(
        &strides[2 * interp_size]);
  }
};

template <int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, scalar_t, index_t, interp_size> {
  static inline bool eval(const int64_t* strides) {
    return true;
  }
};

template <int n, int s, typename scalar_t, typename index_t, int interp_size>
static inline bool check_almost_all_zero_stride(const int64_t* strides) {
  return CheckAlmostAllZeroStrides<n, s, scalar_t, index_t, interp_size>::eval(strides);
}

template <typename scalar_t, typename index_t, int out_ndims, int interp_size>
static inline void basic_loop(char** data, const int64_t* strides, int64_t n) {
  char* dst = data[0];
  char* src = data[1];
  for (int64_t i = 0; i < n; i++) {
    *(scalar_t*)&dst[i * strides[0]] = interpolate<out_ndims, scalar_t, index_t, interp_size>(
        src + i * strides[1], &data[2], &strides[2], i);
  }
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

template <typename scalar_t, typename index_t, int out_ndims, int interp_size>
void ti_cpu_upsample_generic(at::TensorIterator& iter)
{
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {

#ifdef VERBOSE
    if (TI_SHOW_STRIDES) {
      std::cout << "TI_SHOW: N=" << n << std::endl;
      std::cout << "TI_SHOW_STRIDES: "
        << strides[0] << " "
        << strides[1] << " | ";

      int m = 2 * interp_size;

      for (int i=0; i<out_ndims; i++) {
        for (int j=0; j<m; j++) {
          std::cout << strides[m * i + j + 2] << " ";
        }
        std::cout << "| ";
      }
      std::cout << std::endl;
      TI_SHOW_STRIDES = false;
    }
#endif

    // special-cases to let the compiler apply compile-time input-specific optimizations
    if ((strides[0] == sizeof(scalar_t)) && (strides[1] == 0) &&
        check_almost_all_zero_stride<out_ndims, 1, scalar_t, index_t, interp_size>(&strides[2])) {
      // contiguous channels-first case
#ifdef VERBOSE
      if (TI_BASIC_LOOP_CHANNELS_FIRST_TRIGGERED < 1) {
        std::cout << "TI_BASIC_LOOP -> CHANNELS_FIRST" << std::endl << std::flush;
        TI_BASIC_LOOP_CHANNELS_FIRST_TRIGGERED += 1;
      }
#endif
      basic_loop<scalar_t, index_t, out_ndims, interp_size>(data, strides, n);
    } else if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
               check_almost_all_zero_stride<out_ndims, -1, scalar_t, index_t, interp_size>(&strides[2])) {
      // contiguous channels-last case
#ifdef VERBOSE
      if (TI_BASIC_LOOP_CHANNELS_LAST_TRIGGERED < 1) {
        std::cout << "TI_BASIC_LOOP -> CHANNELS_LAST" << std::endl << std::flush;
        TI_BASIC_LOOP_CHANNELS_LAST_TRIGGERED += 1;
      }
#endif
      basic_loop<scalar_t, index_t, out_ndims, interp_size>(data, strides, n);
    } else {
      // fallback
#ifdef VERBOSE
      if (TI_BASIC_LOOP_FALLBACK_TRIGGERED < 1) {
        std::cout << "TI_BASIC_LOOP -> FALLBACK" << std::endl << std::flush;
        TI_BASIC_LOOP_FALLBACK_TRIGGERED += 1;
      }
#endif
      basic_loop<scalar_t, index_t, out_ndims, interp_size>(data, strides, n);
    }
  };

  iter.for_each(loop);
}


template <typename scalar_t, typename index_t, int out_ndims>
void ti_cpu_upsample_generic_aa(at::TensorIterator& iter)
{

  int interp_size = (iter.ntensors() - 2) / out_ndims / 2;

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {

#ifdef VERBOSE
    if (TI_SHOW_STRIDES) {
      std::cout << "AA TI_SHOW: N=" << n << std::endl;
      std::cout << "AA TI_SHOW: interp_size=" << interp_size << std::endl;
      std::cout << "AA TI_SHOW_STRIDES: "
        << strides[0] << " "
        << strides[1] << " | ";

      int m = 2 * interp_size;

      for (int i=0; i<out_ndims; i++) {
        for (int j=0; j<m; j++) {
          std::cout << strides[m * i + j + 2] << " ";
        }
        std::cout << "| ";
      }
      std::cout << std::endl;
      TI_SHOW_STRIDES = false;
    }
#endif

    basic_loop_aa<scalar_t, index_t, out_ndims>(data, strides, n, interp_size);

  };

  iter.for_each(loop);
}


// Helper structs to use with ti_upsample_generic_Nd_kernel_impl
template<typename index_t, typename scalar_t>
struct HelperInterpBase {

  static inline void init_indices_weights(
    std::vector<Tensor> & output, int64_t output_size, int64_t ndims, int64_t reshape_dim, int interp_size
  ) {
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    for (int j=0; j<interp_size; j++) {
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>())));
    }
  }

};

template<typename index_t, typename scalar_t>
struct HelperInterpLinear : public HelperInterpBase<index_t, scalar_t> {

  static const int interp_size = 2;

  // Compute indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -n
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -(n-1)
  //      ...
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  static inline std::vector<Tensor> compute_indices_weights(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, const c10::optional<double> opt_scale, bool antialias
  ) {

    scalar_t scale = area_pixel_compute_scale<scalar_t>(input_size, output_size, align_corners, opt_scale);

    if (antialias && scale > 1.0) {
#ifdef VERBOSE
      std::cout << "-> Antialias option: scale=" << scale << std::endl;
#endif
      return _compute_indices_weights_aa(
        input_size, output_size, stride, ndims, reshape_dim, align_corners, scale
      );
    } else {
      return _compute_indices_weights_no_aa(
        input_size, output_size, stride, ndims, reshape_dim, align_corners, scale
      );
    }
  }

  static inline std::vector<Tensor> _compute_indices_weights_no_aa(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, scalar_t scale
  ) {

    std::vector<Tensor> output;
    HelperInterpLinear<index_t, scalar_t>::init_indices_weights(
      output, output_size, ndims, reshape_dim, HelperInterpLinear<index_t, scalar_t>::interp_size);

    auto input_index0_ptr = output[0].data_ptr<index_t>();
    auto lambda0_ptr = output[1].data_ptr<scalar_t>();
    auto input_index1_ptr = output[2].data_ptr<index_t>();
    auto lambda1_ptr = output[3].data_ptr<scalar_t>();

    for (int64_t i=0; i<output_size; i++) {

      compute_source_index_and_lambda<scalar_t, index_t>(
        input_index0_ptr[i], input_index1_ptr[i],
        lambda0_ptr[i], lambda1_ptr[i],
        scale, i, input_size, output_size, align_corners
      );
      // put stride into indices
      // index values correspond to input indices (0, 1, 2, 3, ...)
      // when multiplied by input stride, maximum possible value
      // input_size[dim-1] * input_size[dim-2] * ... for the given dimension.
      input_index0_ptr[i] *= stride;
      input_index1_ptr[i] *= stride;
    }
    return output;
  }

  // taken from https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29
  static inline scalar_t _filter(scalar_t x) {
      if (x < 0.0) {
          x = -x;
      }
      if (x < 1.0) {
          return 1.0 - x;
      }
      return 0.0;
  }

  static inline std::vector<Tensor> _compute_indices_weights_aa(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, scalar_t scale
  ) {
    TORCH_INTERNAL_ASSERT(scale > 1.0)

    int interp_size = HelperInterpLinear<index_t, scalar_t>::interp_size;
    scalar_t support = (interp_size / 2) * scale;
    interp_size = (int) ceilf(support) * 2 + 1;

    std::vector<Tensor> output;
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // ---- Bounds approach as in PIL -----
    // // bounds: xmin/xmax
    // output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    // output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    // // weights
    // for (int j=0; j<interp_size; j++) {
    //   output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>())));
    // }

    for (int j=0; j<interp_size; j++) {
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>())));
    }

  // ---- Bounds approach as in PIL -----
  //   scalar_t center, total_w, invscale = 1.0 / scale;
  //   index_t zero = static_cast<index_t>(0);
  //   int64_t * idx_ptr_xmin = output[0].data_ptr<index_t>();
  //   int64_t * idx_ptr_xmax = output[1].data_ptr<index_t>();
  //   scalar_t * wt_ptr;

  //   int64_t xmin, xmax, j;

  //   for (int64_t i=0; i<output_size; i++) {

  //     center = scale * (i + 0.5);
  //     xmin = std::max(static_cast<int64_t>(center - support + 0.5), zero);
  //     xmax = std::min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;
  //     idx_ptr_xmin[i] = xmin * stride;
  //     idx_ptr_xmax[i] = xmax * stride;

  //     total_w = 0.0;
  //     for (j=0; j<xmax; j++) {
  //       wt_ptr = output[2 + j].data_ptr<scalar_t>();
  //       scalar_t w = _filter((j + xmin - center + 0.5) * invscale);
  //       wt_ptr[i] = w;
  //       total_w += w;
  //     }
  //     for (j=0; j<xmax; j++) {
  //       wt_ptr = output[2 + j].data_ptr<scalar_t>();
  //       if (total_w != 0.0) {
  //         wt_ptr[i] /= total_w;
  //       }
  //     }
  //     for (; j < interp_size; j++) {
  //       wt_ptr = output[2 + j].data_ptr<scalar_t>();
  //       wt_ptr[i] = static_cast<scalar_t>(0.0);
  //     }
  //   }
  //   return output;
  // }

    scalar_t center, total_w, invscale = 1.0 / scale;
    index_t zero = static_cast<index_t>(0);
    int64_t * idx_ptr;
    scalar_t * wt_ptr;

    int64_t xmin, xmax, j;
    for (int64_t i=0; i<output_size; i++) {

      center = scale * (i + 0.5);
      xmin = std::max(static_cast<int64_t>(center - support + 0.5), zero);
      xmax = std::min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;

      total_w = 0.0;
      for (j=0; j<xmax; j++) {

        idx_ptr = output[2 * j + 0].data_ptr<index_t>();
        idx_ptr[i] = (xmin + j) * stride;

        wt_ptr = output[2 * j + 1].data_ptr<scalar_t>();
        scalar_t w = _filter((j + xmin - center + 0.5) * invscale);
        wt_ptr[i] = w;
        total_w += w;
      }
      for (j=0; j<xmax; j++) {
        wt_ptr = output[2 * j + 1].data_ptr<scalar_t>();
        if (total_w != 0.0) {
          wt_ptr[i] /= total_w;
        }
      }
      for (; j < interp_size; j++) {
        idx_ptr = output[2 * j + 0].data_ptr<index_t>();
        idx_ptr[i] = (xmin + j) * stride;

        wt_ptr = output[2 * j + 1].data_ptr<scalar_t>();
        wt_ptr[i] = static_cast<scalar_t>(0.0);
      }
    }
    return output;
  }

};


template <typename index_t, int out_ndims, typename scale_type, template<typename, typename> class F>
void ti_upsample_generic_Nd_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

#ifdef VERBOSE
  TI_BASIC_LOOP_CHANNELS_FIRST_TRIGGERED = 0;
  TI_BASIC_LOOP_CHANNELS_LAST_TRIGGERED = 0;
  TI_BASIC_LOOP_FALLBACK_TRIGGERED = 0;
  TI_SHOW_STRIDES = true;

  std::cout << "\nInput tensor: " << input.sizes() << std::endl;
  std::cout << "Input is_contiguous memory_format torch.channels_last: " << (input.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
  std::cout << "Input is_contiguous memory_format torch.channels_last_3d: " << (input.is_contiguous(at::MemoryFormat::ChannelsLast3d) ? "true" : "false") << std::endl;
  std::cout << "Input is_contiguous : " << (input.is_contiguous() ? "true" : "false") << std::endl;

  std::cout << "\nOutput tensor: " << output.sizes() << std::endl;
  std::cout << "Output is_contiguous memory_format torch.channels_last: " << (output.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
  std::cout << "Output is_contiguous memory_format torch.channels_last_3d: " << (output.is_contiguous(at::MemoryFormat::ChannelsLast3d) ? "true" : "false") << std::endl;
  std::cout << "Output is_contiguous : " << (output.is_contiguous() ? "true" : "false") << std::endl;
#endif

  // input can be NCHW, NCL or NCKHW
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();
  auto oshape = output.sizes();

  TORCH_INTERNAL_ASSERT(
    shape.size() == oshape.size() && shape.size() == 2 + out_ndims
  );
  TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

  for (int i=0; i<out_ndims; i++) {
    shape[i + 2] = oshape[i + 2];
    strides[i + 2] = 0;
  }
  auto restrided_input = input.as_strided(shape, strides);

  std::vector<std::vector<Tensor>> indices_weights;

  constexpr int interp_size = F<index_t, float>::interp_size;
  auto input_scalar_type = input.scalar_type();

  if (interp_size == 1 && input_scalar_type == at::ScalarType::Byte) {
    // nearest also supports uint8 tensor, but we have to use float
    // with compute_indices_weights
    input_scalar_type = at::ScalarType::Float;
  }

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
    input_scalar_type, "compute_indices_weights_generic", [&] {
      for (int i=0; i<out_ndims; i++) {
        indices_weights.emplace_back(
          F<index_t, scalar_t>::compute_indices_weights(
            input.size(i + 2), oshape[i + 2],
            input.stride(i + 2) * input.element_size(),
            input.dim(), i + 2, align_corners, scales[i],
            antialias
          )
        );
      }
    }
  );

#ifdef VERBOSE
  std::cout << "Size of indices_weights: " << indices_weights.size() << std::endl;
  int counter = 1;
  for (auto & idx_weight: indices_weights) {
    std::cout << "- dim " << counter << " size: " << idx_weight.size() << std::endl;
    counter++;
  }
#endif

  TensorIteratorConfig config;
  config.check_all_same_dtype(false)
    .declare_static_dtype_and_device(input.scalar_type(), input.device())
    .add_output(output)
    .add_input(restrided_input);

  for (auto & idx_weight: indices_weights) {
    for (auto& tensor : idx_weight) {
      config.add_input(tensor);
    }
  }

  auto iter = config.build();

  if (antialias) {
    if (interp_size > 1) {
      // Nearest also supports uint8 tensor, so need to handle it separately
      AT_DISPATCH_FLOATING_TYPES(
          iter.dtype(), "upsample_generic_Nd", [&] {
          ti_cpu_upsample_generic_aa<scalar_t, index_t, out_ndims>(iter);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
          iter.dtype(), "upsample_generic_Nd", [&] {
          ti_cpu_upsample_generic_aa<scalar_t, index_t, out_ndims>(iter);
      });
    }
  } else {
    if (interp_size > 1) {
      // Nearest also supports uint8 tensor, so need to handle it separately
      AT_DISPATCH_FLOATING_TYPES(
          iter.dtype(), "upsample_generic_Nd", [&] {
            constexpr int interp_size = F<index_t, scalar_t>::interp_size;
            ti_cpu_upsample_generic<scalar_t, index_t, out_ndims, interp_size>(iter);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
          iter.dtype(), "upsample_generic_Nd", [&] {
            constexpr int interp_size = F<index_t, scalar_t>::interp_size;
            ti_cpu_upsample_generic<scalar_t, index_t, out_ndims, interp_size>(iter);
      });
    }
  }

}


// Below code is a C++ API for this main.cpp

void _ti_upsample_bilinear2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool antialias) {

#ifndef USE_ALWAYS_INDEX64
  if (canUse32BitIndexMath(input)) {
    ti_upsample_generic_Nd_kernel_impl<int32_t, 2, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_h, scales_w}, antialias);
  } else {
    ti_upsample_generic_Nd_kernel_impl<int64_t, 2, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_h, scales_w}, antialias);
  }
#else
  ti_upsample_generic_Nd_kernel_impl<int64_t, 2, scale_t, HelperInterpLinear>(
    output, input, align_corners, {scales_h, scales_w}, antialias);
#endif
}


Tensor ti_upsample_bilinear2d_cpu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<c10::ArrayRef<double>> scale_factors,
    bool antialias=false) {

  // UpSampleBilinear2d.cpp
  auto output = at::empty({0}, input.options());
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);

  auto full_output_size = native::upsample_2d_common_check(input.sizes(), osize);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  output.resize_(full_output_size, input.suggest_memory_format());
  _ti_upsample_bilinear2d_kernel_impl(output, input, align_corners, scale_h, scale_w, antialias);
  return output;
}


} // anonymous namespace
} // namespace native
} // namespace at

