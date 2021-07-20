import argparse
import os
import numpy as np
import PIL
from PIL import Image
from functools import partial

import torch
from torch.utils.cpp_extension import load
import torch.utils.benchmark as benchmark

from torch.testing._internal.common_utils import gradcheck

# Original image size: 906, 438
sizes = [
    (320, 196),
    (460, 220),
    (120, 96),
    (1200, 196),
    (120, 1200),
]


def pth_downsample(img, mode, size):

    align_corners = False
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img[None, ...].float(), size=size,
        mode=mode,
        align_corners=align_corners,
    )
    return out[0, ...].byte()


def pth_downsample_f32(img, mode, size):

    align_corners = False
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img[None, ...], size=size,
        mode=mode,
        align_corners=align_corners,
    )
    return out[0, ...]


def proto_downsample(img, *, aa_interp, mode, size):
    if mode == "bilinear":
        out = aa_interp.linear_forward(
            img[None, ...].float(),
            size,
            False,
        )
    elif mode == "nearest":
        out = aa_interp.nearest_forward(
            img[None, ...].float(),
            size,
            False,
        )
    elif mode == "bicubic":
        out = aa_interp.cubic_forward(
            img[None, ...].float(),
            size,
            False,
        )
        # apply clip
        out = torch.clamp(out, 0, 255)
    else:
        raise ValueError(mode)
    return out[0, ...].byte()


def proto_downsample_f32(img, *, aa_interp, mode, size):
    if mode == "bilinear":
        out = aa_interp.linear_forward(
            img,
            size,
            False,
        )
    elif mode == "nearest":
        out = aa_interp.nearest_forward(
            img,
            size,
            False,
        )
    elif mode == "bicubic":
        out = aa_interp.cubic_forward(
            img,
            size,
            False,
        )
    else:
        raise ValueError(mode)
    return out


def proto_downsample_f32_backward(aa_interp, mode, grad_output, osize, isize):

    if mode == "bilinear":
        grad_input = aa_interp.linear_backward(
            grad_output,
            osize,
            isize,
            False,
        )
    # elif mode == "bicubic":
    #     out = aa_interp.cubic_backward(
    #         img,
    #         size,
    #         False,
    #     )
    else:
        raise ValueError(mode)

    return grad_input


def get_proto_downsample_function(aa_interp, mode, size):

    class F(torch.autograd.Function):

        interp_module = aa_interp
        interp_mode = mode
        interp_size = size

        @staticmethod
        def forward(ctx, i):
            result = proto_downsample_f32(i, aa_interp=F.interp_module, mode=F.interp_mode, size=F.interp_size)
            ctx.save_for_backward(i, result)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            i, result = ctx.saved_tensors
            ishape = i.shape
            oshape = result.shape[2:]
            return proto_downsample_f32_backward(
                F.interp_module, F.interp_mode, grad_output, oshape, ishape
            )

    return F


class ProtoDownsample(torch.nn.Module):

    def __init__(self, aa_interp, mode, size):
        super().__init__()
        self.downsample_function = get_proto_downsample_function(aa_interp, mode, size)

    def forward(self, x):
        out = self.downsample_function.apply(x[None, ...])
        return out[0, ...]


resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def run_bench(size, mode):
    # All variables are taken from __main__ scope

    inv_size = size[::-1]
    resample = resampling_map[mode]

    mem_format = "channels_last" if t_img.is_contiguous(memory_format=torch.channels_last) else "channels_first"
    is_contiguous = "contiguous" if t_img.is_contiguous() else "non-contiguous"

    label = f"Downsampling: {t_img.shape} -> {size}"
    sub_label = f"{mem_format} {is_contiguous}"
    min_run_time = 3

    results = [
        benchmark.Timer(
            # pil_img.resize(size, resample=resample_val)
            stmt=f"img.resize(size, resample=resample_val)",
            globals={
                "img": pil_img,
                "size": size,
                "resample_val": resample,
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"PIL {PIL.__version__}",
        ).blocked_autorange(min_run_time=min_run_time),

        benchmark.Timer(
            # pth_downsample(t_img, mode, size)
            stmt=f"f(x, mode, size)",
            globals={
                "x": t_img,
                "size": inv_size,
                "mode": mode,
                "f": pth_downsample
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"{torch.version.__version__}",
        ).blocked_autorange(min_run_time=min_run_time),

        benchmark.Timer(
            # proto_downsample
            stmt=f"f(x, aa_interp=op, mode=mode, size=size)",
            globals={
                "x": t_img,
                "mode": mode,
                "size": inv_size,
                "op": aa_interp,
                "f": proto_downsample
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=proto_name,
        ).blocked_autorange(min_run_time=min_run_time * 2),

        benchmark.Timer(
            # proto_downsample_f32
            stmt=f"f(x, aa_interp=op, mode=mode, size=size)",
            globals={
                "x": t_img[None, ...].float(),
                "mode": mode,
                "size": inv_size,
                "op": aa_interp,
                "f": proto_downsample_f32
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"{proto_name} wo float/byte conversion",
        ).blocked_autorange(min_run_time=min_run_time * 2),
    ]
    return results


def run_profiling(size):
    # All variables are taken from __main__ scope

    inv_size = size[::-1]

    t_img_f = t_img[None, ...].float()

    def proto_downsample_forward_only(op, img_f, size):
        out = op.forward(img_f, size, False)
        return out

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, ]) as p:
        proto_downsample_forward_only(aa_interp, t_img_f, inv_size)

    print(p.key_averages().table(row_limit=-1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test interpolation with anti-alias option")
    parser.add_argument(
        "--bench", action="store_true",
        help="Run time benchmark"
    )
    parser.add_argument(
        "--mode", default="bilinear", type=str,
        choices=["bilinear", "nearest", "bicubic"],
        help="Interpolation mode"
    )
    parser.add_argument(
        "--step", default="step_one", type=str,
        choices=["step_zero", "step_one", "step_two", "step_two_dot_one", "step_three", "step_four", "step_two_dot_two"],
        help="Step to use"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Run time profiling"
    )
    parser.add_argument(
        "--flags", choices=["avx", "debug", ],
        help="Define flags according to the choice"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Run time profiling"
    )
    parser.add_argument(
        "--backward", action="store_true",
        help="Test backward op"
    )
    parser.add_argument(
        "--size", type=int, nargs=2,
        help="Use the specified size for the tests"
    )

    args = parser.parse_args()

    if args.bench and args.profile:
        raise RuntimeError("Can not perform profiling and bench together")

    mode = args.mode
    proto_name = f"aa_interp_{mode}_{args.step}"

    if args.cuda:
        sources = [
            f"{args.step}/cuda/extension_interpolate.cpp",
            f"{args.step}/cuda/aa_interpolation_impl.cuh"
        ]
    else:
        sources = [f"{args.step}/extension_interpolate.cpp", ]

    for f in sources:
        assert os.path.exists(f), f"File '{f}' not found"

    if args.flags == "debug":
        extra_cflags = ["-O0", "-g"]
    elif args.flags == "avx":
        extra_cflags = ["-O3", "-mfma", "-mavx", "-mavx2"]
    else:
        extra_cflags = ["-O3", ]

    aa_interp = load(name=proto_name, sources=sources, verbose=True, extra_cflags=extra_cflags)

    pil_img = Image.open("data/test.png").convert("RGB")

    device = "cuda" if args.cuda else "cpu"
    check_backward = args.backward
    resample = resampling_map[mode]

    if args.size is not None:
        print(f"Use specified size: {args.size}")
        sizes = [args.size, ]

    for size in sizes:
        inv_size = size[::-1]
        pil_img_dn = pil_img.resize(size, resample=resample)
        t_pil_img_dn = torch.from_numpy(np.asarray(pil_img_dn).copy().transpose((2, 0, 1))).to(device)

        t_img = torch.from_numpy(np.asarray(pil_img).copy().transpose((2, 0, 1))).contiguous()
        t_img1 = t_img.clone()
        t_img2 = t_img.clone()

        if check_backward:
            t_img1 = t_img1.float()
            t_img1.requires_grad_(True)
            t_img2 = t_img2.float()
            t_img2.requires_grad_(True)
            pth_op = pth_downsample_f32
            proto_downsample_op = ProtoDownsample(aa_interp, mode, inv_size)
        else:
            pth_op = pth_downsample
            proto_downsample_op = partial(proto_downsample, aa_interp=aa_interp, mode=mode, size=inv_size)

        print("mem_format: ", "channels_last" if t_img1.is_contiguous(memory_format=torch.channels_last) else "channels_first")
        print("is_contiguous: ", t_img1.is_contiguous())

        pth_img_dn = pth_op(t_img1, mode, inv_size)
        proto_img_dn = proto_downsample_op(t_img2)

        mae = torch.mean(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
        max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
        print("PyTorch vs PIL: Mean Absolute Error:", mae.item())
        print("PyTorch vs PIL: Max Absolute Error:", max_abs_err.item())

        mae = torch.mean(torch.abs(t_pil_img_dn.float() - proto_img_dn.float()))
        max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - proto_img_dn.float()))
        print("Proto vs PIL: Mean Absolute Error:", mae.item())
        print("Proto vs PIL: Max Absolute Error:", max_abs_err.item())

        if mode == "bilinear":
            assert mae.item() < 1.0
            assert max_abs_err.item() < 1.0 + 1e-5
        elif mode == "nearest":
            pass
            # assert mae.item() < 5.0
            # assert max_abs_err.item() < 1.0 + 1e-5
        elif mode == "bicubic":
            assert mae.item() < 1.0
            assert max_abs_err.item() < 20.0

        if not check_backward:
            proto_pil_dn = Image.fromarray(proto_img_dn.permute(1, 2, 0).numpy())
            fname = f"data/proto_{proto_name}_output_{size[0]}_{size[1]}.png"
            proto_pil_dn.save(fname)
            print(f"Saved downsampled proto output: {fname}")

        if check_backward:
            pth_img_dn.sum().backward()
            print("pth grads", t_img1.grad.mean())
            proto_img_dn.sum().backward()
            print("proto grads", t_img2.grad.mean())

            # grad check
            t_img2 = t_img.clone().double()
            t_img2.requires_grad_(True)
            x = (t_img2, )

            assert gradcheck(proto_downsample_op, x, eps=1e-8, atol=1e-6, rtol=1e-6, check_batched_grad=False)

            # pth_downsample_f32_ = partial(pth_downsample_f32, mode=mode, size=inv_size)
            # assert gradcheck(pth_downsample_f32_, x, eps=1e-8, atol=1e-6, rtol=1e-5, check_batched_grad=False)


    if args.bench:

        if device == "cuda":
            raise NotImplementedError("Currently, do not run any benchmark with cuda")

        print(f"Torch config: {torch.__config__.show()}")
        print(f"Num threads: {torch.get_num_threads()}")

        all_results = []
        for s in sizes:
            all_results += run_bench(s, mode)
        compare = benchmark.Compare(all_results)
        compare.print()

    if args.profile:
        run_profiling(size[0])
