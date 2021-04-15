
#include "aa_interpolation_impl.h"


int main(int argc, char** argv) {

    at::Tensor input = at::rand({1, 3, 1024, 1024});
    int64_t osize[2] = {128, 128};
    c10::optional<at::IntArrayRef> output_size = osize;

    auto res = at::native::ti_upsample::ti_upsample_bilinear2d_cpu(
      input, output_size, false, {}, /*antialias=*/true
    );

    return (int) res.numel();
}