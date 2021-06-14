#include <iostream>
#include <ATen/ATen.h>


namespace playground {

// --------------------------------------------------------------------------------

void test_0() {

    auto output = at::arange(10, at::CUDA(at::kFloat));
    std::cout << "output.device: " << output.device() << std::endl;
    std::cout << "output: " << output << std::endl;

}

// --------------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ static scalar_t bilinear_filter(scalar_t x) {
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return static_cast<scalar_t>(1.0) - x;
  }
  return static_cast<scalar_t>(0.0);
}


template <typename scalar_t>
__device__ __forceinline__ static void _compute_weights(
    const int64_t i,
    const int64_t input_size,
    const scalar_t scale,
    const scalar_t support,
    scalar_t * wt_ptr,
    int64_t interp_size,
    int64_t & xmin,
    int64_t & xmax) {

    scalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    scalar_t center = scale * (i + 0.5);
    xmin = max(static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
    xmax = min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;

    scalar_t total_w = 0.0;
    int64_t j = 0;
    for (j = 0; j < xmax; j++) {
        scalar_t w = bilinear_filter((j + xmin - center + 0.5) * invscale);
        wt_ptr[j] = static_cast<scalar_t>(w);
        total_w += w;
    }
    for (j = 0; j < xmax; j++) {
        if (total_w != 0.0) {
            wt_ptr[j] /= total_w;
        }
    }
    for (; j < interp_size; j++) {
        wt_ptr[j] = static_cast<scalar_t>(0.0);
    }
}


template<typename scalar_t>
__global__ void test_1_kernel(
    const int n,
    at::PackedTensorAccessor64<scalar_t, 1> odata,
    int64_t input_size,
    scalar_t scale,
    scalar_t support
) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index > n) return;

    const int interp_size = (int)ceilf(support) * 2 + 1;

    extern __shared__ int smem[];
    scalar_t * w_ptr = reinterpret_cast<scalar_t*>(smem);

    int64_t xmin, xsize;
    _compute_weights(0, input_size, scale, support, w_ptr, interp_size, xmin, xsize);

    odata[index] = w_ptr[index];

}


void test_1() {

    using scalar_t = float;

    const int64_t input_size = 64;
    const int64_t output_size = 10;
    const scalar_t scale = input_size * 1.0 / output_size;
    int interp_size = 2;
    const scalar_t support = interp_size * 0.5 * scale;
    interp_size = (int)ceilf(support) * 2 + 1;

    auto output = at::empty(interp_size, at::CUDA(at::kFloat));
    auto odata = output.packed_accessor64<scalar_t, 1>();

    size_t shmem_size = (interp_size) * sizeof(scalar_t);

    test_1_kernel<scalar_t>
        <<<1, 256, shmem_size>>>(
            interp_size,
            odata,
            input_size,
            scale,
            support
    );

    auto output_cpu = output.cpu();
    float * o_ptr = (float *) output_cpu.data_ptr();

    std::cout << "output: " << std::endl;
    for (int i=0; i<interp_size; i++) {
        std::cout << o_ptr[i] << " ";
    }
    std::cout << std::endl;

}

// --------------------------------------------------------------------------------


template<typename scalar_t>
__global__ void test_2_kernel(
    const int n,
    at::PackedTensorAccessor64<scalar_t, 1> idata,
    at::PackedTensorAccessor64<scalar_t, 1> odata,
    scalar_t scale,
    scalar_t support
) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index > n) return;

    const int interp_size = (int)ceilf(support) * 2 + 1;
    const int input_size = idata.size(0);

    extern __shared__ int smem[];
    scalar_t * w_ptr = reinterpret_cast<scalar_t*>(smem) + threadIdx.x * interp_size;

    int64_t xmin, xsize;
    _compute_weights(index, input_size, scale, support, w_ptr, interp_size, xmin, xsize);

    scalar_t t = idata[xmin];
    scalar_t wts = w_ptr[0];
    scalar_t output = t * wts;

    int64_t j = 1 ;
    for (; j<xsize; j++) {
      wts = w_ptr[j];
      t = idata[xmin + j];
      output += t * wts;
    }

    odata[index] = output;

}


void test_2() {

    using scalar_t = float;

    const int64_t input_size = 64;
    const int64_t output_size = 10;
    const scalar_t scale = input_size * 1.0 / output_size;
    int interp_size = 2;
    const scalar_t support = interp_size * 0.5 * scale;
    interp_size = (int)ceilf(support) * 2 + 1;

    auto input = at::arange(input_size, at::CUDA(at::kFloat));
    auto idata = input.packed_accessor64<scalar_t, 1>();
    auto output = at::empty(output_size, at::CUDA(at::kFloat));
    auto odata = output.packed_accessor64<scalar_t, 1>();

    size_t n_threads = 256;
    size_t shmem_size = n_threads * (interp_size) * sizeof(scalar_t);

    test_2_kernel<scalar_t>
        <<<1, n_threads, shmem_size>>>(
            output_size,
            idata,
            odata,
            scale,
            support
    );

    auto output_cpu = output.cpu();
    float * o_ptr = (float *) output_cpu.data_ptr();

    std::cout << "output: " << std::endl;
    for (int i=0; i<output_size; i++) {
        std::cout << o_ptr[i] << " ";
    }
    std::cout << std::endl;

}

// --------------------------------------------------------------------------------


template<typename scalar_t>
__global__ void test_3_kernel(
    const int n,
    at::PackedTensorAccessor64<scalar_t, 1> idata,
    at::PackedTensorAccessor64<scalar_t, 1> odata,
    scalar_t scale,
    scalar_t support
) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index > n) return;

    const int interp_size = (int)ceilf(support) * 2 + 1;
    const int input_size = idata.size(0);

    // static instead of dynamic shared memory
    // max supported scale is 127
    scalar_t weights[256];

    int64_t xmin, xsize;
    _compute_weights(index, input_size, scale, support, weights, interp_size, xmin, xsize);

    scalar_t t = idata[xmin];
    scalar_t wts = weights[0];
    scalar_t output = t * wts;

    int64_t j = 1 ;
    for (; j<xsize; j++) {
      wts = weights[j];
      t = idata[xmin + j];
      output += t * wts;
    }

    odata[index] = output;

}


void test_3() {

    using scalar_t = float;

    const int64_t input_size = 64;
    const int64_t output_size = 10;
    const scalar_t scale = input_size * 1.0 / output_size;
    int interp_size = 2;
    const scalar_t support = interp_size * 0.5 * scale;
    interp_size = (int)ceilf(support) * 2 + 1;

    auto input = at::arange(input_size, at::CUDA(at::kFloat));
    auto idata = input.packed_accessor64<scalar_t, 1>();
    auto output = at::empty(output_size, at::CUDA(at::kFloat));
    auto odata = output.packed_accessor64<scalar_t, 1>();

    size_t n_threads = 256;
    test_3_kernel<scalar_t>
        <<<1, n_threads>>>(
            output_size,
            idata,
            odata,
            scale,
            support
    );

    auto output_cpu = output.cpu();
    float * o_ptr = (float *) output_cpu.data_ptr();

    std::cout << "output: " << std::endl;
    for (int i=0; i<output_size; i++) {
        std::cout << o_ptr[i] << " ";
    }
    std::cout << std::endl;

}

// --------------------------------------------------------------------------------

}