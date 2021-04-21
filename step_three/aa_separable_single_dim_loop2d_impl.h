
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
static int TI_BASIC_LOOP_ZERO_STRIDES_TRIGGERED = 0;
static int TI_BASIC_LOOP_NONZERO_STRIDES_TRIGGERED = 0;
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

template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim_zero_strides(char* src, char** data, int64_t i, const index_t ids_stride) {

  index_t ids_min = *(index_t*)&data[0][0];
  index_t ids_size = *(index_t*)&data[1][0];
  // index_t ids_stride = *(index_t*)&data[2][0];

  scalar_t t = *(scalar_t *)&src[ids_min];
  index_t wts_idx = *(index_t*)&data[4][0];
  scalar_t wts = *(scalar_t*)&data[3][wts_idx];

  scalar_t output = t * wts;
  for (int j=1; j<ids_size; j++) {
    wts = *(scalar_t*)&data[3][wts_idx + j * sizeof(scalar_t)];
    t = *(scalar_t *)&src[ids_min + j * ids_stride];
    output += t * wts;
  }
  return output;
}

template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim(char* src, char** data, const int64_t* strides, int64_t i, const index_t ids_stride) {

  index_t ids_min = *(index_t*)&data[0][i * strides[0]];
  index_t ids_size = *(index_t*)&data[1][i * strides[1]];
  // index_t ids_stride = *(index_t*)&data[2][i * strides[2]];

  scalar_t t = *(scalar_t *)&src[ids_min];
  index_t wts_idx = *(index_t*)&data[4][i * strides[4]];
  scalar_t wts = *(scalar_t*)&data[3][wts_idx];

  scalar_t output = t * wts;
  for (int j=1; j<ids_size; j++) {
    wts = *(scalar_t*)&data[3][wts_idx + j * sizeof(scalar_t)];
    t = *(scalar_t *)&src[ids_min + j * ids_stride];
    output += t * wts;
  }
  return output;
}


template <typename scalar_t, typename index_t>
static inline void basic_loop_aa_single_dim_zero_strides(char** data, const int64_t* strides, int64_t size0, int64_t size1) {

  constexpr int ntensor = 2 + 3 + 2;  // output, input, 3 indices, 2 weights
  const int64_t* outer_strides = &strides[ntensor];
  char *src, *dst;
  // index stride is constant for the given dimension
  const index_t ids_stride = *(index_t*)&data[2 + 2][0];

  for (int64_t j=0; j < size1; j++) {
    if (j > 0) {
      for (int arg = 0; arg < ntensor; arg++) {
        data[arg] += outer_strides[arg];
      }
    }
    dst = data[0];
    src = data[1];

    for (int64_t i = 0; i < size0; i++) {
      *(scalar_t*)&dst[i * strides[0]] = interpolate_aa_single_dim_zero_strides<scalar_t, index_t>(
          src + i * strides[1], &data[2], i, ids_stride);
    }
  }
}

template <typename scalar_t, typename index_t>
static inline void basic_loop_aa_single_dim_nonzero_strides(char** data, const int64_t* strides, int64_t size0, int64_t size1) {

  constexpr int ntensor = 2 + 3 + 2;  // output, input, 3 indices, 2 weights
  const int64_t* outer_strides = &strides[ntensor];
  char *src, *dst;
  // index stride is constant for the given dimension
  const index_t ids_stride = *(index_t*)&data[2 + 2][0];

  for (int64_t j=0; j < size1; j++) {

    if (j > 0) {
      for (int arg = 0; arg < 2; arg++) {
        data[arg] += outer_strides[arg];
      }
    }
    dst = data[0];
    src = data[1];

    for (int64_t i = 0; i < size0; i++) {
      *(scalar_t*)&dst[i * strides[0]] = interpolate_aa_single_dim<scalar_t, index_t>(
          src, &data[2], &strides[2], i, ids_stride);
    }
  }
}


template <typename scalar_t, typename index_t>
static inline void basic_loop_aa_single_dim_fallback(char** data, const int64_t* strides, int64_t size0, int64_t size1) {

  constexpr int ntensor = 2 + 3 + 2;  // output, input, 3 indices, 2 weights
  const int64_t* outer_strides = &strides[ntensor];
  char *src, *dst;
  // index stride is constant for the given dimension
  const index_t ids_stride = *(index_t*)&data[2 + 2][0];

  for (int64_t j=0; j < size1; j++) {

    if (j > 0) {
      for (int arg = 0; arg < ntensor; arg++) {
        data[arg] += outer_strides[arg];
      }
    }
    dst = data[0];
    src = data[1];

    for (int64_t i = 0; i < size0; i++) {
      *(scalar_t*)&dst[i * strides[0]] = interpolate_aa_single_dim<scalar_t, index_t>(
          src + i * strides[1], &data[2], &strides[2], i, ids_stride);
    }
  }
}


template <typename scalar_t, typename index_t, int out_ndims>
void ti_cpu_upsample_generic_aa_single_dim(at::TensorIterator& iter, int interp_size=-1)
{

  auto loop2d = [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {

#ifdef VERBOSE
    TORCH_INTERNAL_ASSERT(interp_size > 0);
    if (TI_SHOW_STRIDES) {
      std::cout << "AA TI_SHOW: size0=" << size0 << std::endl;
      std::cout << "AA TI_SHOW: size1=" << size1 << std::endl;
      std::cout << "AA TI_SHOW: interp_size=" << interp_size << std::endl;
      std::cout << "AA TI_SHOW_STRIDES: "
        << strides[0] << " "
        << strides[1] << " | ";

      constexpr int m = 3 + 2;
      int ndims = 1;
      for (int i=0; i<ndims; i++) {
        for (int j=0; j<m; j++) {
          std::cout << strides[m * i + j + 2] << " ";
        }
        std::cout << "| ";
      }
      std::cout << std::endl;

      int ntensor = iter.ntensors();
      const int64_t* outer_strides = &strides[ntensor];
      std::cout << " - strides= ";
      for (int64_t arg = 0; arg < ntensor; arg++) {
        std::cout << strides[arg] << " ";
      }
      std::cout << std::endl;
      std::cout << " - outer_strides= ";
      for (int64_t arg = 0; arg < ntensor; arg++) {
          std::cout << outer_strides[arg] << " ";
      }
      std::cout << std::endl;

      TI_SHOW_STRIDES = false;
    }
#endif

    constexpr int ntensor = 2 + 3 + 2;  // output, input, 3 indices, 2 weights
    const int64_t* outer_strides = &strides[ntensor];

    if (
      (strides[0] == sizeof(scalar_t))
      && (strides[1] == sizeof(scalar_t))
      && is_zero_stride<3 + 2>(&strides[2])
    )
    {
  #ifdef VERBOSE
      if (TI_BASIC_LOOP_ZERO_STRIDES_TRIGGERED < 1) {
        std::cout << "AA TI_BASIC_LOOP -> ZERO_STRIDES" << std::endl << std::flush;
        TI_BASIC_LOOP_ZERO_STRIDES_TRIGGERED += 1;
      }
  #endif
      basic_loop_aa_single_dim_zero_strides<scalar_t, index_t>(data, strides, size0, size1);
    }
    else if (
      (strides[0] == sizeof(scalar_t))
      && (strides[1] == 0)
      && is_zero_stride<3 + 2>(&outer_strides[2])
    )
    {
  #ifdef VERBOSE
      if (TI_BASIC_LOOP_NONZERO_STRIDES_TRIGGERED < 1) {
        std::cout << "AA TI_BASIC_LOOP -> NONZERO_STRIDES" << std::endl << std::flush;
        TI_BASIC_LOOP_NONZERO_STRIDES_TRIGGERED += 1;
      }
  #endif
      basic_loop_aa_single_dim_nonzero_strides<scalar_t, index_t>(data, strides, size0, size1);
    }
    else
    {
  #ifdef VERBOSE
      if (TI_BASIC_LOOP_FALLBACK_TRIGGERED < 1) {
        std::cout << "AA TI_BASIC_LOOP -> FALLBACK" << std::endl << std::flush;
        TI_BASIC_LOOP_FALLBACK_TRIGGERED += 1;
      }
  #endif
      basic_loop_aa_single_dim_fallback<scalar_t, index_t>(data, strides, size0, size1);
    }

  };

  iter.for_each(loop2d);
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

    // ---- Bounds approach as in PIL -----
    // bounds: xmin/xmax
    output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));

    {
      // Weights
      new_shape[reshape_dim] = output_size * interp_size;
      auto wts = empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>()));
      auto strides = wts.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      wts = wts.as_strided(new_shape, strides);
      output.emplace_back(wts);
      // Weights indices
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    }

    scalar_t center, total_w, invscale = 1.0 / scale;
    index_t zero = static_cast<index_t>(0);
    int64_t * idx_ptr_xmin = output[0].data_ptr<index_t>();
    int64_t * idx_ptr_size = output[1].data_ptr<index_t>();
    int64_t * idx_ptr_stride = output[2].data_ptr<index_t>();
    scalar_t * wt_ptr = output[3].data_ptr<scalar_t>();
    int64_t * wt_idx_ptr = output[4].data_ptr<index_t>();

    int64_t xmin, xmax, j;

    for (int64_t i=0; i<output_size; i++) {

      center = scale * (i + 0.5);
      xmin = std::max(static_cast<int64_t>(center - support + 0.5), zero);
      xmax = std::min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;
      idx_ptr_xmin[i] = xmin * stride;
      idx_ptr_size[i] = xmax;
      idx_ptr_stride[i] = stride;

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
void _ti_separable_upsample_generic_Nd_kernel_impl_single_dim(
    Tensor& output,
    const Tensor& input,
    int interp_dim,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

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
  }
  strides[interp_dim] = 0;
  auto restrided_input = input.as_strided(shape, strides);

  std::vector<std::vector<Tensor>> indices_weights;

  int interp_size = F<index_t, float>::interp_size;
  auto input_scalar_type = input.scalar_type();

  if (interp_size == 1 && input_scalar_type == at::ScalarType::Byte) {
    // nearest also supports uint8 tensor, but we have to use float
    // with compute_indices_weights
    input_scalar_type = at::ScalarType::Float;
  }

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
    input_scalar_type, "compute_indices_weights_generic", [&] {

      indices_weights.emplace_back(
        F<index_t, scalar_t>::compute_indices_weights(
          input.size(interp_dim), oshape[interp_dim],
          input.stride(interp_dim) * input.element_size(),
          input.dim(), interp_dim, align_corners, scales[interp_dim - 2],
          antialias, interp_size
        )
      );
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
          ti_cpu_upsample_generic_aa_single_dim<scalar_t, index_t, out_ndims>(iter, interp_size);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte,
          iter.dtype(), "upsample_generic_Nd", [&] {
          ti_cpu_upsample_generic_aa_single_dim<scalar_t, index_t, out_ndims>(iter, interp_size);
      });
    }
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }

}


template <typename index_t, int out_ndims, typename scale_type, template<typename, typename> class F>
void ti_separable_upsample_generic_Nd_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

#ifdef VERBOSE
  TI_BASIC_LOOP_ZERO_STRIDES_TRIGGERED = 0;
  TI_BASIC_LOOP_NONZERO_STRIDES_TRIGGERED = 0;
  TI_BASIC_LOOP_FALLBACK_TRIGGERED = 0;
  TI_SHOW_STRIDES = true;


  std::cout << "\n--- ti_separable_upsample_generic_Nd_kernel_impl: "<< std::endl;
  std::cout << "\nInput tensor: " << input.sizes() << std::endl;
  std::cout << "Input is_contiguous memory_format torch.channels_last: " << (input.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
  std::cout << "Input is_contiguous memory_format torch.channels_last_3d: " << (input.is_contiguous(at::MemoryFormat::ChannelsLast3d) ? "true" : "false") << std::endl;
  std::cout << "Input is_contiguous : " << (input.is_contiguous() ? "true" : "false") << std::endl;

  std::cout << "\nOutput tensor: " << output.sizes() << std::endl;
  std::cout << "Output is_contiguous memory_format torch.channels_last: " << (output.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
  std::cout << "Output is_contiguous memory_format torch.channels_last_3d: " << (output.is_contiguous(at::MemoryFormat::ChannelsLast3d) ? "true" : "false") << std::endl;
  std::cout << "Output is_contiguous : " << (output.is_contiguous() ? "true" : "false") << std::endl;
#endif

  auto temp_oshape = input.sizes().vec();
  at::Tensor temp_output, temp_input = input;
  for (int i=0; i<out_ndims-1; i++) {
    int interp_dim = 2 + out_ndims - 1 - i;
    temp_oshape[interp_dim] = output.sizes()[interp_dim];
    temp_output = at::empty(temp_oshape);
    #ifdef VERBOSE
      std::cout << temp_input.sizes() << "->" << temp_output.sizes() << std::endl;
    #endif
    _ti_separable_upsample_generic_Nd_kernel_impl_single_dim<index_t, out_ndims, scale_t, HelperInterpLinear>(
      temp_output, temp_input, interp_dim, align_corners, scales, antialias
    );
    temp_input = temp_output;
  }

  #ifdef VERBOSE
    std::cout << temp_input.sizes() << "->" << output.sizes() << std::endl;
    TI_SHOW_STRIDES = true;
    TI_BASIC_LOOP_ZERO_STRIDES_TRIGGERED = 0;
    TI_BASIC_LOOP_NONZERO_STRIDES_TRIGGERED = 0;
    TI_BASIC_LOOP_FALLBACK_TRIGGERED = 0;
  #endif
  _ti_separable_upsample_generic_Nd_kernel_impl_single_dim<index_t, out_ndims, scale_t, HelperInterpLinear>(
    output, temp_input, 2, align_corners, scales, antialias
  );
}

} // anonymous namespace
} // namespace native
} // namespace at

