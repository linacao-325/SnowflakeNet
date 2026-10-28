import torch
from experiments.pointnet_diffusion import Experiment, GlobalEncoder

torch.set_num_threads(2)
torch.manual_seed(7)
p, gt = torch.randn(2, 32, 3), torch.randn(2, 64, 3)
m = Experiment(timesteps=20, refine=False)
f = m.teacher(p, gt)
assert f.shape == (2, 256)
assert torch.allclose(f, m.teacher(p[:, torch.randperm(32)], gt[:, torch.randperm(64)]), atol=2e-6)
condition, _ = m.condition(p, gt)
loss, _, _ = m.diffusion.loss(gt[:, :16], condition)
loss.backward()
assert any(x.grad is not None for x in m.teacher.parameters())
m.zero_grad(set_to_none=True); m.set_stage('student'); m.train()
assert not m.teacher.training
condition, kd = m.condition(p, gt)
loss, _, _ = m.diffusion.loss(gt[:, :16], condition)
(loss + kd).backward()
assert all(x.grad is None and not x.requires_grad for x in m.teacher.parameters())
assert any(x.grad is not None for x in m.student.parameters())
assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
m.eval()
x = m(p, steps=4, generator=torch.Generator().manual_seed(1))[0]
y = m(p, steps=4, generator=torch.Generator().manual_seed(1))[0]
assert x.shape == (2, 256, 3) and torch.isfinite(x).all()
assert torch.equal(x, y)
try:
    m.diffusion.sample(condition, steps=21)
    raise AssertionError('invalid steps accepted')
except ValueError:
    pass
print('PASS: pooling invariance, teacher/student gradients, frozen teacher, deterministic finite DDIM sampling')
