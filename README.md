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



## Refs:

- https://github.com/pytorch/vision/issues/2950

- https://tcapelle.github.io/capeblog/pytorch/fastai/2021/02/26/image_resizing.html

- https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3

- https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/src/libImaging/Resample.c#L196-L203