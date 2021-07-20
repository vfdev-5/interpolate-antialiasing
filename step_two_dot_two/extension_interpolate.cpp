#include <torch/extension.h>

#include "aa_interpolation_impl.h"
#include "aa_interpolation_backward_impl.h"


torch::Tensor interpolate_linear_forward(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

    return at::native::ti_upsample::ti_upsample_bilinear2d_cpu(
      input, output_size, align_corners, {}, /*antialias=*/true
    );
}

torch::Tensor interpolate_linear_backward(const torch::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners=false) {

    return at::native::ti_upsample::ti_upsample_bilinear2d_backward_cpu(
      grad_output, output_size, input_size, align_corners, {}, /*antialias=*/true
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



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_forward", &interpolate_linear_forward, "Anti-Aliased Linear Interpolation forward");
  m.def("nearest_forward", &interpolate_nearest_forward, "Anti-Aliased Nearest Interpolation forward");  // it's not nearest but box
  m.def("cubic_forward", &interpolate_cubic_forward, "Anti-Aliased Cubic Interpolation forward");
  m.def("linear_backward", &interpolate_linear_backward, "Anti-Aliased Linear Interpolation forward");
}