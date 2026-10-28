"""Exact Chamfer L1 with bounded-memory nearest-neighbor search."""
import torch


def chamfer(a, b, one_way=False):
    def directed(x, y):
        # Search without autograd: retaining every cdist block for backward is
        # quadratic in point count even when the forward computation is chunked.
        indices = []
        with torch.no_grad():
            for source in x.detach().float().split(256, dim=1):
                best = torch.full(source.shape[:2], float('inf'), device=x.device)
                nearest = torch.zeros(source.shape[:2], dtype=torch.long, device=x.device)
                offset = 0
                for target in y.detach().float().split(1024, dim=1):
                    distance, index = torch.cdist(source, target).min(-1)
                    update = distance < best
                    nearest = torch.where(update, index + offset, nearest)
                    best = torch.minimum(best, distance)
                    offset += target.shape[1]
                indices.append(nearest)
        index = torch.cat(indices, dim=1)
        matched = y.gather(1, index.unsqueeze(-1).expand(-1, -1, 3))
        # Only matched point pairs are retained for backward. Piecewise exact
        # Chamfer gradient, with first-index tie breaking.
        return torch.linalg.vector_norm(x.float() - matched.float(), dim=-1).mean()

    ab = directed(a, b)
    return ab if one_way else (ab + directed(b, a)) / 2
