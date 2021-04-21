# Prototype Torch Interpolate with anti-aliasing

Problem: see [notebooks/analysis.ipynb](notebooks/analysis.ipynb)

## TL;DR:

Currently:

- MAE(downsampled_pil, downsampled_torch) >> 1
- MaxAbsE(downsampled_pil, downsampled_torch) > 100

We would like:

- MAE(downsampled_pil, downsampled_torch) ~ 1
- MaxAbsE(downsampled_pil, downsampled_torch) < 10


## Algorithm ([PIL implementation](https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/src/libImaging/Resample.c#L196-L203))


For 1D, output pixel is computed as

```
output[ox]  = input[xmin + 0] * kernel[x + 0]
output[ox] += input[xmin + 1] * kernel[x + 1]
output[ox] += input[xmin + 2] * kernel[x + 2]
...
output[ox] += input[xmin + n] * kernel[x + n]
```
where `n = ceil(support * scale) * 2 + 1` and
```
support = 1  # for bilinear
support = 2  # for bicubic
scale = input_size / output_size

center = (ox + 0.5) * scale
xmin = max( round(center - support), 0 )
```

Kernel values are computed using [triangle filtering (bilinear mode)](https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/src/libImaging/Resample.c#L20-L29)
```
kernel[x + k] = triangle(...)
```

## Step 0


<details>

<summary>
Result 1 : cxxflag by default and non-separable version
</summary>

```bash
PYTHONPATH=/pytorch/ python test.py

Input tensor: [1, 3, 438, 906]
Input is_contiguous memory_format torch.channels_last: true
Input is_contiguous memory_format torch.channels_last_3d: false
Input is_contiguous : false

Output tensor: [1, 3, 196, 320]
Output is_contiguous memory_format torch.channels_last: false
Output is_contiguous memory_format torch.channels_last_3d: false
Output is_contiguous : true
-> Antialias option: scale=2.23469
-> Antialias option: scale=2.83125
Size of indices_weights: 2
- dim 1 size: 14
- dim 2 size: 14
AA TI_SHOW: N=320
AA TI_SHOW: interp_size=7
AA TI_SHOW_STRIDES: 4 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 8 4 8 4 8 4 8 4 8 4 8 4 8 4 |
PyTorch vs PIL: Mean Absolute Error: 6.302572250366211
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5034226179122925
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_s0_output.png
```

```bash
OMP_NUM_THREADS=6 PYTHONPATH=/pytorch/ python test.py --bench

PyTorch vs PIL: Mean Absolute Error: 6.302572250366211
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5034226179122925
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_s0_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 6
[---------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) -----------]
                      |  PIL 8.1.2  |  1.9.0a0+git8518b0e  |  aa_interp_lin_s0
6 threads: -------------------------------------------------------------------
      channels_first  |     2.0     |         1.2          |        10.2

Times are in milliseconds (ms).
```

</details>


<details>

<summary>
Result 2 : cxxflag: `-O3` and separable version
</summary>

We are using PIL-SIMD here

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/pytorch/ python test.py --bench

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_s0_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 1
[------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) -------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_s0
1 threads: ------------------------------------------------------------------------------------
      channels_first contiguous  |       350.6       |        668.4         |       5630.3

Times are in microseconds (us).
```

```bash
OMP_NUM_THREADS=6 PYTHONPATH=/pytorch/ python test.py --bench

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_s0_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 6
[------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) -------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_s0
6 threads: ------------------------------------------------------------------------------------
      channels_first contiguous  |       339.9       |        153.6         |       1123.4

Times are in microseconds (us).
```

</details>


## Step 1


<details>

<summary>
Result 1 : cxxflag: `-O3` and separable version, indices as bounds
</summary>

We are using PIL-SIMD here

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/pytorch/ python test.py --bench --step=step_one

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0Saved downsampled proto output: data/proto_aa_interp_lin_step_one_output.pngTorch config: PyTorch built with:  - GCC 9.3  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK
-O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 1
[---------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) ----------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_one
1 threads: ------------------------------------------------------------------------------------------
      channels_first contiguous  |       351.2       |        666.6         |          3376.9

Times are in microseconds (us).
```

```bash
OMP_NUM_THREADS=6 PYTHONPATH=/pytorch/ python test.py --bench --step=step_one

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_one_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 6
[---------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) ----------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_one
6 threads: ------------------------------------------------------------------------------------------
      channels_first contiguous  |       345.7       |        155.0         |          989.0

Times are in microseconds (us).
```

</details>


## Step 2


<details>

<summary>
Result : cxxflag: `-O3` and separable version, indices as bounds, single weights tensor
</summary>

We are using PIL-SIMD here

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/pytorch/ python test.py --bench --step=step_two

mem_format:  channels_firstis_contiguous:  TruePyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_two_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF,
USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 1
[---------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) ----------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_two
1 threads: ------------------------------------------------------------------------------------------
      channels_first contiguous  |       345.9       |        670.7         |          2927.0

Times are in microseconds (us).
```

```bash
OMP_NUM_THREADS=6 PYTHONPATH=/pytorch/ python test.py --bench --step=step_two

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_two_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 6
[---------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) ----------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_two
6 threads: ------------------------------------------------------------------------------------------
      channels_first contiguous  |       341.7       |        154.1         |          574.9

Times are in microseconds (us).
```

</details>


## Step 2.1

<details>

<summary>
Result : cxxflag: `-O3` and separable version, indices as bounds, single weights tensor + optim tricks
</summary>

We are using PIL-SIMD here

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/pytorch/ python test.py --bench --step=step_two_dot_one

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_two_dot_one_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 1
[-------------------------------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) --------------------------------------------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_two_dot_one  |  aa_interp_lin_step_two_dot_one wo float/byte conversion
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       333.3       |        659.9         |              2500.4              |                           2175.3

Times are in microseconds (us).
```

</details>


## Step 3

<details>

<summary>
Result : cxxflag: `-O3` and single TI version with indices as bounds, single weights tensor + optim tricks
</summary>

We are using PIL-SIMD here

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/pytorch/ python test.py --bench --step=step_three

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.302402019500732
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035501718521118
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_two_dot_one_output.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 1
[-------------------------------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) --------------------------------------------------------]
                                 |  PIL 7.0.0.post3  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_two_dot_one  |  aa_interp_lin_step_two_dot_one wo float/byte conversion
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       335.7       |        661.3         |              3069.2              |                           2744.7

Times are in microseconds (us).
```

</details>



## Step 2.2

<details>

<summary>
Result : cxxflag: `-O3` and separable version, indices as bounds, single weights tensor + more optim tricks
</summary>

We are using Pillow without SIMD

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/pytorch/ python test.py --bench --step=step_two_dot_two

mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 6.3022003173828125
PyTorch vs PIL: Max Absolute Error: 151.0
Proto vs PIL: Mean Absolute Error: 0.5035820603370667
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_two_dot_two_output_320_196.png
mem_format:  channels_first
is_contiguous:  True
PyTorch vs PIL: Mean Absolute Error: 13.175492286682129
PyTorch vs PIL: Max Absolute Error: 172.0
Proto vs PIL: Mean Absolute Error: 0.5021122694015503
Proto vs PIL: Max Absolute Error: 1.0
Saved downsampled proto output: data/proto_aa_interp_lin_step_two_dot_two_output_120_96.png
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON,

Num threads: 1
[----------------------------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) -----------------------------------------------------]
                                 |  PIL 8.2.0  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_two_dot_two  |  aa_interp_lin_step_two_dot_two wo float/byte conversion
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |    1842.8   |        664.0         |              2272.6              |                           1938.4

Times are in microseconds (us).

[------------------------------------------------------ Downsampling: torch.Size([3, 438, 906]) -> (120, 96) -----------------------------------------------------]
                                 |  PIL 8.2.0  |  1.9.0a0+gitb5647dd  |  aa_interp_lin_step_two_dot_two  |  aa_interp_lin_step_two_dot_two wo float/byte conversion
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |    1296.6   |        350.2         |              1918.2              |                           1674.0

Times are in microseconds (us).
```

</details>


## Refs:

- https://github.com/pytorch/vision/issues/2950

- https://tcapelle.github.io/capeblog/pytorch/fastai/2021/02/26/image_resizing.html

- https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3

- https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/src/libImaging/Resample.c#L196-L203



## Development

<details>

<summary>
Docker container setup
</summary>

```bash
docker run --rm -it \
    --name=tv-interpolate \
    -v $PWD:/interpolate-antialiasing \
    -w /interpolate-antialiasing \
    --network=host --security-opt seccomp:unconfined --privileged --shm-size 16G \
    nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04 \
    /bin/bash
```

```
apt-get update && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y git cmake python3 python3-pip numactl && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    pip install numpy typing_extensions Pillow ninja
```

- Install linux perf

```
echo "deb http://archive.ubuntu.com/ubuntu/ bionic main universe\n" >> /etc/apt/sources.list && \
apt-get update && apt-get install -y linux-tools-4.15.0-20-generic linux-tools-4.15.0-20 linux-tools-4.15.0-20-lowlatency && \
rm -rf /usr/bin/perf && \
ln -s /usr/lib/linux-tools-4.15.0-20/perf /usr/bin/perf
```

- Install Pillow-SIMD

```
apt-get install -y libpng-dev libjpeg-turbo8-dev
pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

</details>