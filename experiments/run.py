"""Run from repository root: python -m experiments.run --help."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
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
    from models.utils import fps_subsample
    cfg = read_yaml(args.config)
    if cfg.dataset.name != 'PCN':
        p.error('this runner currently supports PCN only')
    device = torch.device(args.device)
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
    (out / 'arguments.json').write_text(json.dumps(vars(args), indent=2))

    def loader(split):
        return torch.utils.data.DataLoader(PCNDataLoader(cfg).get_dataset(split),
            batch_size=cfg.train.batch_size if split == 'train' else cfg.test.batch_size,
            shuffle=split == 'train', num_workers=cfg.train.num_workers,
            collate_fn=collate_fn)

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
        result = evaluate(loader(args.split))
        (out / 'metrics.json').write_text(json.dumps(result, indent=2)); print(result)
        return
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=.5)
    train = loader('train')
    # Stage 1 has GT-conditioned features and cannot give student validation metrics.
    validation = loader('val') if args.stage == 'student' else None
    best = float('inf')
    for epoch in range(args.epochs):
        model.train(); sums = np.zeros(3); count = 0
        for _, _, batch in train:
            partial, gt = batch['partial_cloud'].to(device), batch['gtcloud'].to(device)
            condition, kd = model.condition(partial, gt)
            target = fps_subsample(gt, 256)
            diffusion, _, _ = model.diffusion.loss(target, condition)
            # Train SPD on low-noise x0 estimates; full noise range trains epsilon.
            _, seeds, _ = model.diffusion.loss(target, condition, max_t=100)
            outputs = model.refine(seeds, partial, condition)
            completion = sum(chamfer(x, fps_subsample(gt, x.shape[1])) for x in outputs[:-1])
            completion = completion + chamfer(outputs[-1], gt) + chamfer(partial, outputs[-1], True)
            loss = diffusion + completion + args.kd_weight * kd
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            sums += np.array([diffusion.item(), completion.item(), kd.item()]) * len(gt)
            count += len(gt)
        if not count: raise RuntimeError('empty train split')
        scheduler.step()
        record = dict(epoch=epoch+1, diffusion=sums[0]/count, completion=sums[1]/count, kd=sums[2]/count)
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


if __name__ == '__main__':
    main()
