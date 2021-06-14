#include <torch/extension.h>

#include "aa_interpolation_impl.cuh"


torch::Tensor interpolate_linear_forward_cuda(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

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

    return at::native::aa_interpolation::upsample_bilinear2d_out_cuda_template(
      output, input, output_size, align_corners
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_forward_cuda", &interpolate_linear_forward_cuda, "Anti-Aliased Linear Interpolation forward on CUDA");
}