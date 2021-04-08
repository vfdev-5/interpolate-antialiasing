#include <torch/extension.h>

#include "aa_interpolation_impl.h"


torch::Tensor interpolate_linear_forward(const torch::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners=false) {

    return at::native::ti_upsample::ti_upsample_bilinear2d_cpu(
      input, output_size, align_corners, {}, /*antialias=*/true
    );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &interpolate_linear_forward, "Anti-Aliasing Interpolationg forward");
}