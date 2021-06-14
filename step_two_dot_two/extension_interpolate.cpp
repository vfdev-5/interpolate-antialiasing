#include <torch/extension.h>

#include "aa_interpolation_impl.h"


torch::Tensor interpolate_linear_forward(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

    return at::native::ti_upsample::ti_upsample_bilinear2d_cpu(
      input, output_size, align_corners, {}, /*antialias=*/true
    );

}

torch::Tensor interpolate_nearest_forward(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

    return at::native::ti_upsample::ti_upsample_nearest2d_cpu(
      input, output_size, align_corners, {}, /*antialias=*/true
    );
}

torch::Tensor interpolate_cubic_forward(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

    return at::native::ti_upsample::ti_upsample_bicubic2d_cpu(
      input, output_size, align_corners, {}, /*antialias=*/true
    );
}


torch::Tensor interpolate_linear_forward_cuda(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

    return at::native::aa_interpolation::upsample_bilinear2d_out_cuda_template(
      input, output_size, align_corners, {}, /*antialias=*/true
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_forward", &interpolate_linear_forward, "Anti-Aliased Linear Interpolation forward");
  m.def("nearest_forward", &interpolate_nearest_forward, "Anti-Aliased Nearest Interpolation forward");
  m.def("cubic_forward", &interpolate_cubic_forward, "Anti-Aliased Cubic Interpolation forward");
  m.def("linear_forward_cuda", &interpolate_linear_forward_cuda, "Anti-Aliased Linear Interpolation forward on CUDA");
}