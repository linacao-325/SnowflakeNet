"""Run from repository root: python -m experiments.run --help."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from experiments.distributed import setup, TrainingObjective
from experiments.pointnet_diffusion import Experiment


def chamfer(a, b, one_way=False):
    # Exact chunked Euclidean CD-L1, same averaging convention as baseline.
    def directed(x, y):
        return torch.cat([torch.cdist(z, y).amin(-1) for z in x.split(256, 1)], 1).mean()
    ab = directed(a, b)
    return ab if one_way else (ab + directed(b, a)) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='completion/configs/pcn_cd1.yaml')
    p.add_argument('--stage', choices=['teacher', 'student', 'eval'], required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--out', default='work/pointnet_diffusion')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--seed', type=int, default=2022)
    p.add_argument('--kd-weight', type=float, default=1.)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.stage != 'teacher' and not args.checkpoint:
        p.error('student/eval require --checkpoint')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from completion.utils.yaml_reader import read_yaml
    from completion.core.datasets.pcn import PCNDataLoader, collate_fn
    cfg = read_yaml(args.config)
    if cfg.dataset.name != 'PCN':
        p.error('this runner currently supports PCN only')
    device, rank, world = setup(args.device)
    primary = rank == 0
    model = Experiment().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if args.stage == 'student' and checkpoint['stage'] != 'teacher':
            p.error('student initialization requires a teacher checkpoint')
        if args.stage == 'eval' and checkpoint['stage'] != 'student':
            p.error('reported evaluation requires a student checkpoint')
        model.load_state_dict(checkpoint['model'])
    model.set_stage('teacher' if args.stage == 'teacher' else 'student')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if primary:
        (out / 'arguments.json').write_text(json.dumps(vars(args), indent=2))

    def loader(split):
        dataset = PCNDataLoader(cfg).get_dataset(split)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     seed=args.seed, shuffle=True) if split == 'train' and world > 1 else None
        return torch.utils.data.DataLoader(dataset,
            batch_size=cfg.train.batch_size if split == 'train' else cfg.test.batch_size,
            shuffle=split == 'train' and sampler is None, sampler=sampler,
            num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    @torch.no_grad()
    def evaluate(data):
        model.eval()
        total, count, seconds = 0., 0, 0.
        generator = torch.Generator(device=device).manual_seed(args.seed)
        for _, _, batch in data:
            partial, gt = batch['partial_cloud'].to(device), batch['gtcloud'].to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model(partial, steps=args.steps, generator=generator)[-1]
            if device.type == 'cuda': torch.cuda.synchronize()
            seconds += time.perf_counter() - start
            total += chamfer(prediction, gt).item() * len(gt); count += len(gt)
        if not count: raise RuntimeError('empty evaluation split')
        return {'cd_l1_x1000': total / count * 1000, 'seconds_per_shape': seconds / count,
                'shapes': count, 'steps': args.steps, 'timing': 'includes first batch, excludes data loading and CD'}

    if args.stage == 'eval':
        if primary:
            result = evaluate(loader(args.split))
            (out / 'metrics.json').write_text(json.dumps(result, indent=2)); print(result)
        if world > 1:
            dist.barrier(); dist.destroy_process_group()
        return
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=.5)
    objective = TrainingObjective(model, args.kd_weight).to(device)
    if world > 1:
        objective = DDP(objective, device_ids=[device.index] if device.type == 'cuda' else None)
    # DDP synchronized initialization; use different per-rank training noise.
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    train = loader('train')
    # Stage 1 has GT-conditioned features and cannot give student validation metrics.
    validation = loader('val') if args.stage == 'student' and primary else None
    best = float('inf')
    for epoch in range(args.epochs):
        objective.train(); sums = np.zeros(3); count = 0
        if isinstance(train.sampler, DistributedSampler):
            train.sampler.set_epoch(epoch)
        for _, _, batch in train:
            partial, gt = batch['partial_cloud'].to(device), batch['gtcloud'].to(device)
            loss, components = objective(partial, gt)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            sums += components.cpu().numpy() * len(gt)
            count += len(gt)
        totals = torch.tensor([*sums, count], device=device, dtype=torch.float64)
        if world > 1: dist.all_reduce(totals)
        sums = totals[:3].cpu().numpy(); count = int(totals[3].item())
        if not count: raise RuntimeError('empty train split')
        scheduler.step()
        record = dict(epoch=epoch+1, diffusion=sums[0]/count, completion=sums[1]/count, kd=sums[2]/count)
        # Rank 0 evaluates the complete split without sampler padding or duplicates.
        if world > 1: dist.barrier()
        if not primary:
            dist.barrier()
            continue
        score = None
        if validation is not None:
            record.update(evaluate(validation)); score = record['cd_l1_x1000']
        checkpoint = dict(model=model.state_dict(), stage=args.stage, epoch=epoch+1,
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), args=vars(args))
        torch.save(checkpoint, out / 'last.pt')
        if score is not None and score < best:
            best = score; torch.save(checkpoint, out / 'best.pt')
        with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(record, flush=True)
        if world > 1: dist.barrier()
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
