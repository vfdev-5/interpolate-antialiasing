import argparse
import numpy as np
import PIL
from PIL import Image

import torch
from torch.utils.cpp_extension import load
import torch.utils.benchmark as benchmark

sizes = [
    (320, 196),
    (120, 96),
]


def pth_downsample(img, size):
    out = torch.nn.functional.interpolate(
        img[None, ...].float(), size=size,
        mode="bilinear",
        align_corners=False,
    )
    return out[0, ...].byte()


def proto_downsample(aa_interp_lin, img, size):
    out = aa_interp_lin.forward(
        img[None, ...].float(),
        size,
        False,
    )
    return out[0, ...].byte()


def proto_downsample_f32(aa_interp_lin, img, size):
    out = aa_interp_lin.forward(
        img,
        size,
        False,
    )
    return out[0, ...]


def run_bench(size):
    # All variables are taken from __main__ scope

    inv_size = size[::-1]

    mem_format = "channels_last" if t_img.is_contiguous(memory_format=torch.channels_last) else "channels_first"
    is_contiguous = "contiguous" if t_img.is_contiguous() else "non-contiguous"

    label = f"Downsampling: {t_img.shape} -> {size}"
    sub_label = f"{mem_format} {is_contiguous}"
    min_run_time = 3

    results = [
        benchmark.Timer(
            # pil_img.resize(size, resample=2)
            stmt=f"img.resize(size, resample=2)",
            globals={
                "img": pil_img,
                "size": size,
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"PIL {PIL.__version__}",
        ).blocked_autorange(min_run_time=min_run_time),

        benchmark.Timer(
            # pth_downsample(t_img, size)
            stmt=f"f(x, size)",
            globals={
                "x": t_img,
                "size": inv_size,
                "f": pth_downsample
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"{torch.version.__version__}",
        ).blocked_autorange(min_run_time=min_run_time),

        benchmark.Timer(
            # proto_downsample(aa_interp_lin, t_img, size)
            stmt=f"f(op, x, size)",
            globals={
                "x": t_img,
                "size": inv_size,
                "op": aa_interp_lin,
                "f": proto_downsample
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=proto_name,
        ).blocked_autorange(min_run_time=min_run_time * 2),

        benchmark.Timer(
            # proto_downsample_f32(aa_interp_lin, t_img, size)
            stmt=f"f(op, x, size)",
            globals={
                "x": t_img[None, ...].float(),
                "size": inv_size,
                "op": aa_interp_lin,
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

    def proto_downsample_forward_only(aa_interp_lin, img_f, size):
        out = aa_interp_lin.forward(img_f, size, False)
        return out

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, ]) as p:
        proto_downsample_forward_only(aa_interp_lin, t_img_f, inv_size)

    print(p.key_averages().table(row_limit=-1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test interpolation with anti-alias option")
    parser.add_argument(
        "--bench", action="store_true",
        help="Run time benchmark"
    )
    parser.add_argument(
        "--step", default="step_one", type=str,
        choices=["step_zero", "step_one", "step_two", "step_two_dot_one", "step_three", "step_four", "step_two_dot_two"],
        help="Run time benchmark"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Run time profiling"
    )

    args = parser.parse_args()

    if args.bench and args.profile:
        raise RuntimeError("Can not perform profiling and bench together")

    proto_name = f"aa_interp_lin_{args.step}"
    proto_src = f"{args.step}/extension_interpolate.cpp"
    # extra_cflags = ["-O3", ]
    extra_cflags = ["-O3", "-mfma", "-mavx", "-mavx2"]
    aa_interp_lin = load(name=proto_name, sources=[proto_src], verbose=True, extra_cflags=extra_cflags)

    pil_img = Image.open("data/test.png").convert("RGB")

    for size in sizes:
        inv_size = size[::-1]
        pil_img_dn = pil_img.resize(size, resample=2)
        t_pil_img_dn = torch.from_numpy(np.asarray(pil_img_dn).copy().transpose((2, 0, 1)))

        t_img = torch.from_numpy(np.asarray(pil_img).copy().transpose((2, 0, 1))).contiguous()
        print("mem_format: ", "channels_last" if t_img.is_contiguous(memory_format=torch.channels_last) else "channels_first")
        print("is_contiguous: ", t_img.is_contiguous())

        pth_img_dn = pth_downsample(t_img, inv_size)
        proto_img_dn = proto_downsample(aa_interp_lin, t_img, inv_size)

        mae = torch.mean(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
        max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
        print("PyTorch vs PIL: Mean Absolute Error:", mae.item())
        print("PyTorch vs PIL: Max Absolute Error:", max_abs_err.item())

        mae = torch.mean(torch.abs(t_pil_img_dn.float() - proto_img_dn.float()))
        max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - proto_img_dn.float()))
        print("Proto vs PIL: Mean Absolute Error:", mae.item())
        print("Proto vs PIL: Max Absolute Error:", max_abs_err.item())

        proto_pil_dn = Image.fromarray(proto_img_dn.permute(1, 2, 0).numpy())
        fname = f"data/proto_{proto_name}_output_{size[0]}_{size[1]}.png"
        proto_pil_dn.save(fname)
        print(f"Saved downsampled proto output: {fname}")

    if args.bench:

        print(f"Torch config: {torch.__config__.show()}")
        print(f"Num threads: {torch.get_num_threads()}")

        all_results = []
        for s in sizes:
            all_results += run_bench(s)
        compare = benchmark.Compare(all_results)
        compare.print()

    if args.profile:
        run_profiling(size[0])
