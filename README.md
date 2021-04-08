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

Kernel values are computed as following
```

```

## Step 0

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


## Refs:

- https://github.com/pytorch/vision/issues/2950

- https://tcapelle.github.io/capeblog/pytorch/fastai/2021/02/26/image_resizing.html

- https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3

- https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/src/libImaging/Resample.c#L196-L203