"""CPU integration test; substitutes FPS/SPD, exercises the real DDP objective."""
import sys
import types
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from experiments.distributed import setup, TrainingObjective
from experiments.pointnet_diffusion import Experiment


def main():
    torch.set_num_threads(1)
    device, rank, world = setup('cpu')
    assert world == 2
    stub = types.ModuleType('models.utils')
    stub.fps_subsample = lambda x, n: x[:, :n]
    sys.modules['models.utils'] = stub
    for stage in ('teacher', 'student'):
        torch.manual_seed(3)
        model = Experiment(refine=False)
        model.set_stage(stage)
        model.refine = lambda seeds, partial, condition: [seeds]
        wrapped = DDP(TrainingObjective(model, 1.), broadcast_buffers=False)
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=.001)
        torch.manual_seed(17 + rank)
        for _ in range(2):
            loss, metrics = wrapped(torch.randn(2, 32, 3), torch.randn(2, 256, 3))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            assert torch.isfinite(loss)
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, 0)
            assert torch.equal(reference, parameter)
        if stage == 'student':
            assert all(p.grad is None for p in model.teacher.parameters())
    dist.barrier()
    if rank == 0: print('PASS: two ranks, both stages, repeated DDP backward, synchronized weights, frozen teacher')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
