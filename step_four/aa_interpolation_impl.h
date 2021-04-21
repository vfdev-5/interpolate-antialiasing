
#include <cmath>
#include <vector>
#include <ATen/TypeDefault.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/UpSample.h>

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


template <int m>
static inline bool is_zero_stride(const int64_t* strides) {
  bool output = strides[0] == 0;
  for (int i=1; i<m; i++) {
    output &= (strides[i] == 0);
  }
  return output;
}


template <typename data_type, int m>
static inline bool is_contiguous_stride(const int64_t* strides) {
  bool output = strides[0] == sizeof(data_type);
  for (int i=1; i<m; i++) {
    output &= strides[i] == sizeof(data_type);
  }
  return output;
}


template <int n, typename scalar_t, typename index_t>
struct InterpolateAA {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {

      const index_t ids_idx = *(index_t*)&data[0][i * strides[0]];
      const index_t ids_min = *(index_t*)&data[1][ids_idx];
      const index_t ids_size = *(index_t*)&data[1][ids_idx + sizeof(index_t)];
      const index_t ids_stride = *(index_t*)&data[1][ids_idx + sizeof(index_t) + sizeof(index_t)];

      char * src_min = src + ids_min;

      scalar_t t = InterpolateAA<n - 1, scalar_t, index_t>::eval(
          src_min, &data[2 + 2], &strides[2 + 2], i);

      const index_t wts_idx = *(index_t*)&data[2][i * strides[2]];
      char * wts_ptr = &data[3][wts_idx];
      scalar_t wts = *(scalar_t*)&wts_ptr[0];
      scalar_t output = t * wts;

      index_t j = 1;
      for (; j < 2; j++) {
        wts = *(scalar_t*)&wts_ptr[j * sizeof(scalar_t)];
        t = InterpolateAA<n - 1, scalar_t, index_t>::eval(
            src + ids_min + j * ids_stride, &data[2 + 2], &strides[2 + 2], i);
        output += t * wts;
      }

      for (; j<ids_size; j++) {
        wts = *(scalar_t*)&wts_ptr[j * sizeof(scalar_t)];
        t = InterpolateAA<n - 1, scalar_t, index_t>::eval(
            src_min + j * ids_stride, &data[2 + 2], &strides[2 + 2], i);
        output += t * wts;
      }

      return output;
  }
};

template <typename scalar_t, typename index_t>
struct InterpolateAA<1, scalar_t, index_t> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {

      const index_t ids_idx = *(index_t*)&data[0][i * strides[0]];
      const index_t ids_min = *(index_t*)&data[1][ids_idx];
      const index_t ids_size = *(index_t*)&data[1][ids_idx + sizeof(index_t)];
      const index_t ids_stride = *(index_t*)&data[1][ids_idx + sizeof(index_t) + sizeof(index_t)];

      char * src_min = src + ids_min;

      scalar_t t = *(scalar_t *)&src_min[0];
      const index_t wts_idx = *(index_t*)&data[2][i * strides[2]];
      char * wts_ptr = &data[3][wts_idx];
      scalar_t wts = *(scalar_t*)&wts_ptr[0];
      scalar_t output = t * wts;

      index_t j = 1;
      for (; j < 2; j++) {
        wts = *(scalar_t*)&wts_ptr[j * sizeof(scalar_t)];
        t = *(scalar_t *)&src[ids_min + j * ids_stride];
        output += t * wts;
      }

      for (; j<ids_size; j++) {
        wts = *(scalar_t*)&wts_ptr[j * sizeof(scalar_t)];
        t = *(scalar_t *)&src_min[j * ids_stride];
        output += t * wts;
      }
      return output;
    }
};

template <int n, typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa(char* src, char** data, const int64_t* strides, int64_t i) {
  return InterpolateAA<n, scalar_t, index_t>::eval(src, data, strides, i);
}


template <int N, int non_zero_stride_dim, typename index_t>
struct CheckAlmostAllZeroStrides {
  static inline bool eval(const int64_t* strides) {
    // N is dim index: N -> dim0, N-1 -> dim1, ...
    // non_zero_stride_dim should be out_ndims - dim
    bool output;
    if (N == non_zero_stride_dim) {
      output = ((strides[1] == 0) && (strides[0] == sizeof(index_t)));
      output &= ((strides[3] == 0) && (strides[2] == sizeof(index_t)));
    } else {
      output = is_zero_stride<2 + 2>(strides);
    }
    return output &&
      CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, index_t>::eval(
        &strides[2 + 2]);
  }
};

template <int non_zero_stride_dim, typename index_t>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, index_t> {
  static inline bool eval(const int64_t* strides) {
    return true;
  }
};

template <int n, int s, typename index_t>
static inline bool check_almost_all_zero_stride(const int64_t* strides) {
  return CheckAlmostAllZeroStrides<n, s, index_t>::eval(strides);
}

template <typename scalar_t, typename index_t, int out_ndims>
static inline void basic_loop_aa(char** data, const int64_t* strides, int64_t n) {
  char* dst = data[0];
  char* src = data[1];
  for (int64_t i = 0; i < n; i++) {
    *(scalar_t*)&dst[i * strides[0]] = interpolate_aa<out_ndims, scalar_t, index_t>(
        src + i * strides[1], &data[2], &strides[2], i);
  }
}


template <typename scalar_t, typename index_t, int out_ndims>
void ti_cpu_upsample_generic_aa(at::TensorIterator& iter, const int interp_sizes[out_ndims])
{

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {

#ifdef VERBOSE
    if (TI_SHOW_STRIDES) {
      std::cout << "AA TI_SHOW: n=" << n << std::endl;

      for (int i=0; i<out_ndims; i++) {
        std::cout << "AA TI_SHOW: interp_size[" << i << "] = " << interp_sizes[i] << std::endl;
      }
      std::cout << "AA TI_SHOW_STRIDES: "
        << strides[0] << " "
        << strides[1] << " | ";

      constexpr int m = 2 + 2;
      for (int i=0; i<out_ndims; i++) {
        for (int j=0; j<m; j++) {
          std::cout << strides[m * i + j + 2] << " ";
        }
        std::cout << "| ";
      }
      std::cout << std::endl;

      // int ntensor = iter.ntensors();
      // const int64_t* outer_strides = &strides[ntensor];
      // std::cout << " - outer_strides= ";
      // for (int64_t arg = 0; arg < ntensor; arg++) {
      //     std::cout << outer_strides[arg] << " ";
      // }
      // std::cout << std::endl;

      TI_SHOW_STRIDES = false;
    }
#endif

    // special-cases to let the compiler apply compile-time input-specific optimizations
    if (
      (strides[0] == sizeof(scalar_t))
      && (strides[1] == 0)
      // Check if strides for 2 indices and 2 weights are zeros for all dimensions except out_dims - 1
      && check_almost_all_zero_stride<out_ndims, 1, index_t>(&strides[2]))
    {
      // contiguous channels-first case
#ifdef VERBOSE
      if (TI_BASIC_LOOP_CHANNELS_FIRST_TRIGGERED < 1) {
        std::cout << "AA TI_BASIC_LOOP -> CHANNELS_FIRST" << std::endl << std::flush;
        TI_BASIC_LOOP_CHANNELS_FIRST_TRIGGERED += 1;
      }
#endif
      basic_loop_aa<scalar_t, index_t, out_ndims>(data, strides, n);
    }
//     else if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
//                check_almost_all_zero_stride<out_ndims, -1, scalar_t, index_t, interp_size>(&strides[2])) {
//       // contiguous channels-last case
// #ifdef VERBOSE
//       if (TI_BASIC_LOOP_CHANNELS_LAST_TRIGGERED < 1) {
//         std::cout << "AA TI_BASIC_LOOP -> CHANNELS_LAST" << std::endl << std::flush;
//         TI_BASIC_LOOP_CHANNELS_LAST_TRIGGERED += 1;
//       }
// #endif
//       basic_loop<scalar_t, index_t, out_ndims, interp_size>(data, strides, n);
//     }
    else
    {
      // fallback
#ifdef VERBOSE
      if (TI_BASIC_LOOP_FALLBACK_TRIGGERED < 1) {
        std::cout << "AA TI_BASIC_LOOP -> FALLBACK" << std::endl << std::flush;
        TI_BASIC_LOOP_FALLBACK_TRIGGERED += 1;
      }
#endif
      basic_loop_aa<scalar_t, index_t, out_ndims>(data, strides, n);
    }

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

  static inline std::vector<Tensor> compute_indices_weights(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, const c10::optional<double> opt_scale, bool antialias, int & out_interp_size
  ) {

    scalar_t scale = area_pixel_compute_scale<scalar_t>(input_size, output_size, align_corners, opt_scale);

    if (antialias && scale > 1.0) {
#ifdef VERBOSE
      std::cout << "-> Antialias option: scale=" << scale << std::endl;
#endif
      return _compute_indices_weights_aa(
        input_size, output_size, stride, ndims, reshape_dim, align_corners, scale, out_interp_size
      );
    } else {
      TORCH_INTERNAL_ASSERT(false)
    }
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
    bool align_corners, scalar_t scale, int & out_interp_size
  ) {
    TORCH_INTERNAL_ASSERT(scale > 1.0)

    int interp_size = HelperInterpLinear<index_t, scalar_t>::interp_size;
    scalar_t support = (interp_size / 2) * scale;
    interp_size = (int) ceilf(support) * 2 + 1;

    // return interp_size
    out_interp_size = interp_size;

    std::vector<Tensor> output;
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // ---- Bounds approach (as in PIL) in a single tensor -----
    // bounds for indices: indexer and xmin/max
    {
      // Indexer:
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
      // Bounds:
      new_shape[reshape_dim] = output_size * 3; // 3 <=> xmin, size, stride
      auto indices = empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>()));
      auto strides = indices.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      indices = indices.as_strided(new_shape, strides);
      output.emplace_back(indices);
    }

    {
      // Weights indices
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
      // Weights
      new_shape[reshape_dim] = output_size * interp_size;
      auto wts = empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>()));
      auto strides = wts.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      wts = wts.as_strided(new_shape, strides);
      output.emplace_back(wts);
    }

    scalar_t center, total_w, invscale = 1.0 / scale;
    index_t zero = static_cast<index_t>(0);

    int64_t * idx_idx_ptr = output[0].data_ptr<index_t>();
    int64_t * idx_ptr_xmin_size_stride = output[1].data_ptr<index_t>();

    int64_t * wt_idx_ptr = output[2].data_ptr<index_t>();
    scalar_t * wt_ptr = output[3].data_ptr<scalar_t>();

    int64_t xmin, xmax, j;

    for (int64_t i=0; i<output_size; i++) {

      idx_idx_ptr[i] = i * 3 * sizeof(index_t);

      center = scale * (i + 0.5);
      xmin = std::max(static_cast<int64_t>(center - support + 0.5), zero);
      xmax = std::min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;
      idx_ptr_xmin_size_stride[3 * i] = xmin * stride;
      idx_ptr_xmin_size_stride[3 * i + 1] = xmax;
      idx_ptr_xmin_size_stride[3 * i + 2] = stride;

      wt_idx_ptr[i] = i * interp_size * sizeof(scalar_t);

      total_w = 0.0;
      for (j=0; j<xmax; j++) {
        scalar_t w = _filter((j + xmin - center + 0.5) * invscale);
        wt_ptr[i * interp_size + j] = w;
        total_w += w;
      }
      for (j=0; j<xmax; j++) {
        if (total_w != 0.0) {
          wt_ptr[i * interp_size + j] /= total_w;
        }
      }

      for (; j < interp_size; j++) {
        wt_ptr[i * interp_size + j] = static_cast<scalar_t>(0.0);
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

  int out_interp_sizes[out_ndims];

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
    input_scalar_type, "compute_indices_weights_generic", [&] {
      for (int i=0; i<out_ndims; i++) {
        indices_weights.emplace_back(
          F<index_t, scalar_t>::compute_indices_weights(
            input.size(i + 2), oshape[i + 2],
            input.stride(i + 2) * input.element_size(),
            input.dim(), i + 2, align_corners, scales[i],
            antialias, out_interp_sizes[i]
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
          ti_cpu_upsample_generic_aa<scalar_t, index_t, out_ndims>(iter, out_interp_sizes);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
          iter.dtype(), "upsample_generic_Nd", [&] {
          ti_cpu_upsample_generic_aa<scalar_t, index_t, out_ndims>(iter, out_interp_sizes);
      });
    }
  } else {
    TORCH_INTERNAL_ASSERT(false);
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

  ti_upsample_generic_Nd_kernel_impl<int64_t, 2, scale_t, HelperInterpLinear>(
    output, input, align_corners, {scales_h, scales_w}, antialias);

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

