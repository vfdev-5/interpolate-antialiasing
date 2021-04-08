import numpy as np
from PIL import Image

import torch
from torch.utils.cpp_extension import load

size = (320, 196)


def pth_downsample(img, size):
    out = torch.nn.functional.interpolate(
        img[None, ...].float(), size=size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    return out[0, ...].byte()


def proto_downsample(aa_interp_lin, img, size):
    size = size[::-1]
    out = aa_interp_lin.forward(
        img[None, ...].float(),
        size,
        False,
    )
    return out[0, ...].byte()


if __name__ == "__main__":


    proto_name = "aa_interp_lin_s0"
    proto_src = "step_zero/extension_interpolate.cpp"
    aa_interp_lin = load(name=proto_name, sources=[proto_src], verbose=True)

    pil_img = Image.open("data/test.png").convert("RGB")
    pil_img_dn = pil_img.resize(size, resample=2)
    t_pil_img_dn = torch.from_numpy(np.asarray(pil_img_dn).copy().transpose((2, 0, 1)))

    t_img = torch.from_numpy(np.asarray(pil_img).copy().transpose((2, 0, 1)))
    pth_img_dn = pth_downsample(t_img, size)

    proto_img_dn = proto_downsample(aa_interp_lin, t_img, size)

    mae = torch.mean(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
    max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
    print("PyTorch vs PIL: Mean Absolute Error:", mae.item())
    print("PyTorch vs PIL: Max Absolute Error:", max_abs_err.item())

    mae = torch.mean(torch.abs(t_pil_img_dn.float() - proto_img_dn.float()))
    max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - proto_img_dn.float()))
    print("Proto vs PIL: Mean Absolute Error:", mae.item())
    print("Proto vs PIL: Max Absolute Error:", max_abs_err.item())

    proto_pil_dn = Image.fromarray(proto_img_dn.permute(1, 2, 0).numpy())
    proto_pil_dn.save(f"data/proto_{proto_name}_output.png")
    print(f"Saved downsampled proto output: data/proto_{proto_name}_output.png")
