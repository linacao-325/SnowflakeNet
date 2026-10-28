"""Shared single-process / torchrun distributed runtime."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch import nn


def setup(device_name):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        if device_name.startswith('cuda'):
            torch.cuda.set_device(local)
            device = torch.device('cuda', local)
            backend = 'nccl'
        else:
            device, backend = torch.device(device_name), 'gloo'
        dist.init_process_group(backend=backend, timeout=timedelta(hours=4))
        return device, dist.get_rank(), world
    return torch.device(device_name), 0, 1


class TrainingObjective(nn.Module):
    # All trainable paths must run inside DDP.forward, rather than through .module.
    def __init__(self, model, kd_weight):
        super().__init__()
        self.model = model
        self.kd_weight = kd_weight

    def forward(self, partial, gt):
        from models.utils import fps_subsample
        from experiments.losses import chamfer
        condition, kd = self.model.condition(partial, gt)
        # Reuse a nested GT pyramid, matching the baseline's FPS hierarchy.
        with torch.no_grad():
            gt_2048 = fps_subsample(gt, 2048)
            gt_512 = fps_subsample(gt_2048, 512)
            target = fps_subsample(gt_512, 256)
        targets = {256: target, 512: gt_512, 2048: gt_2048}
        diffusion, _, _ = self.model.diffusion.loss(target, condition)
        _, seeds, _ = self.model.diffusion.loss(target, condition, max_t=100)
        outputs = self.model.refine(seeds, partial, condition)
        completion = sum(chamfer(x, targets[x.shape[1]]) for x in outputs[:-1])
        completion = completion + chamfer(outputs[-1], gt) + chamfer(partial, outputs[-1], True)
        loss = diffusion + completion + self.kd_weight * kd
        return loss, torch.stack((diffusion.detach(), completion.detach(), kd.detach()))
