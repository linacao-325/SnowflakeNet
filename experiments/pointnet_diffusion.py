"""PointNet teacher/student conditioning + diffusion seeds + SnowflakeNet SPD."""
import math
import torch
from torch import nn
from torch.nn import functional as fn


class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 128),
                                    nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
        self.pool = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.LayerNorm(256))

    def forward(self, partial, gt=None):
        tagged = [torch.cat((partial, torch.zeros_like(partial[..., :1])), -1)]
        if gt is not None:
            tagged.append(torch.cat((gt, torch.ones_like(gt[..., :1])), -1))
        h = self.points(torch.cat(tagged, 1))
        return self.pool(torch.cat((h.amax(1), h.mean(1)), -1))


class Denoiser(nn.Module):
    """Permutation-equivariant point MLP with global context and FiLM conditioning."""
    def __init__(self, width=256, depth=4):
        super().__init__()
        self.input = nn.Linear(3, width)
        self.time = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))
        self.blocks = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.films = nn.ModuleList([nn.Linear(512, width * 2) for _ in range(depth)])
        self.output = nn.Linear(width, 3)

    def forward(self, x, t, condition):
        freq = torch.exp(-math.log(10000) * torch.arange(32, device=x.device) / 32)
        phase = t[:, None].float() * freq
        context = torch.cat((condition, self.time(torch.cat((phase.sin(), phase.cos()), -1))), -1)
        h = self.input(x)
        for block, norm, film in zip(self.blocks, self.norms, self.films):
            scale, shift = film(context).chunk(2, -1)
            z = block(torch.cat((h, h.amax(1, keepdim=True).expand_as(h)), -1))
            h = h + fn.silu(norm(z) * (1 + scale[:, None]) + shift[:, None])
        return self.output(h)


class CoarseDiffusion(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.net = Denoiser()
        s = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        a = torch.cos((s + .008) / 1.008 * math.pi / 2).square()
        beta = (1 - a[1:] / a[:-1]).clamp(max=.999)
        self.register_buffer('alpha', (1 - beta).cumprod(0).float())

    def loss(self, target, condition, max_t=None):
        t = torch.randint(max_t or len(self.alpha), (len(target),), device=target.device)
        a = self.alpha[t, None, None]
        noise = torch.randn_like(target)
        xt = a.sqrt() * target + (1 - a).sqrt() * noise
        eps = self.net(xt, t, condition)
        x0 = (xt - (1 - a).sqrt() * eps) / a.sqrt()
        return fn.mse_loss(eps, noise), x0, t

    @torch.no_grad()
    def sample(self, condition, n=256, steps=50, generator=None):
        if not 1 <= steps <= len(self.alpha):
            raise ValueError('sampling steps must be in [1, timesteps]')
        x = torch.randn(len(condition), n, 3, device=condition.device, generator=generator)
        times = torch.linspace(len(self.alpha)-1, 0, steps, device=x.device).long().tolist()
        for i, t in enumerate(times):
            a = self.alpha[t]
            eps = self.net(x, torch.full((len(x),), t, device=x.device), condition)
            x0 = (x - (1-a).sqrt() * eps) / a.sqrt()
            previous = self.alpha[times[i+1]] if i+1 < len(times) else x.new_tensor(1.)
            x = previous.sqrt() * x0 + (1-previous).sqrt() * eps
        return x


class Experiment(nn.Module):
    def __init__(self, timesteps=1000, refine=True):
        super().__init__()
        self.teacher = GlobalEncoder()
        self.student = GlobalEncoder()
        self.diffusion = CoarseDiffusion(timesteps)
        self.stage = 'teacher'
        self.refine_enabled = refine
        if refine:
            from models.SPD import SPD
            self.uppers = nn.ModuleList([SPD(dim_feat=256, up_factor=f, i=i) for i, f in enumerate([1, 4, 8])])
        self.set_stage('teacher')

    def set_stage(self, stage):
        if stage not in ('teacher', 'student'):
            raise ValueError(stage)
        self.stage = stage
        for p in self.teacher.parameters():
            p.requires_grad_(stage == 'teacher')
        for p in self.student.parameters():
            p.requires_grad_(stage == 'student')
        if stage == 'student':
            self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 'student':
            self.teacher.eval()
        return self

    def condition(self, partial, gt):
        if self.stage == 'teacher':
            return self.teacher(partial, gt), partial.new_zeros(())
        with torch.no_grad():
            teacher = self.teacher(partial, gt)
        student = self.student(partial)
        return student, fn.mse_loss(student, teacher)

    def refine(self, seeds, partial, condition):
        from models.utils import fps_subsample
        p = fps_subsample(torch.cat((seeds, partial), 1), 512).transpose(1, 2).contiguous()
        outputs, previous = [seeds], None
        for upper in self.uppers:
            p, previous = upper(p, condition.unsqueeze(-1), previous)
            outputs.append(p.transpose(1, 2).contiguous())
        return outputs

    @torch.no_grad()
    def forward(self, partial, steps=50, generator=None):
        # GT is deliberately absent from the inference API.
        condition = self.student(partial)
        seeds = self.diffusion.sample(condition, steps=steps, generator=generator)
        return self.refine(seeds, partial, condition) if self.refine_enabled else [seeds]
