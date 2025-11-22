

import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image


@torch.no_grad()
def show_prediction_fm_fid(
    valloader,
    fm_scheduler,     
    model,
    device,
    inference_step=50,
    out_dir="checkpoints/fm/val_samples",
    eps=1e-8
):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    save_idx = 0

    timestep = torch.linspace(0, 1 - (1e-8), inference_step, device=device)

    for img, cls in tqdm(valloader):
        img = img.to(device)
        real_imgs = img.detach().cpu()
        for b in range(real_imgs.size(0)):
            save_image(real_imgs[b], os.path.join(real_dir, f"{save_idx:06d}.png"))
            save_idx += 1

        img_scaled = img * 2 - 1 
        B = img_scaled.size(0)

        x_t = torch.randn_like(img_scaled)

        for k, t in enumerate(timestep):
            if k < inference_step - 1:
                dt = timestep[k+1] - timestep[k]
            else:
                dt = timestep[k] - timestep[k-1]
                
            t_batch = torch.full((B,), float(t.item()), device=device)

            v = model(x_t, t_batch) 
            x_t = x_t + dt * v
        gen_imgs = (x_t + 1) / 2

        start = save_idx - B
        for b in range(B):
            save_image(gen_imgs[b].cpu(), os.path.join(gen_dir, f"{start+b:06d}.png"))

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir