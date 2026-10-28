"""Exact Chamfer L1 with bounded-memory nearest-neighbor search."""
import importlib
import importlib.util
from pathlib import Path
import torch


_BACKEND = 'auto'
_CUDA_SEARCH = None


def configure_chamfer(backend='auto'):
    global _BACKEND
    if backend not in ('auto', 'cuda', 'chunked'):
        raise ValueError(backend)
    _BACKEND = backend


def _load_cuda_search():
    global _CUDA_SEARCH
    if _CUDA_SEARCH is not None:
        return _CUDA_SEARCH
    # Reuse Simba's installed extension when present.
    if importlib.util.find_spec('chamfer') is not None:
        extension = importlib.import_module('chamfer')
        _CUDA_SEARCH = extension.forward
        print('Chamfer backend: installed Simba CUDA extension', flush=True)
    else:
        # Load the file directly: importing loss_functions would also load EMD.
        path = Path(__file__).resolve().parents[1] / 'loss_functions/Chamfer3D/dist_chamfer_3D.py'
        spec = importlib.util.spec_from_file_location('_snowflake_chamfer_cuda', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CUDA_SEARCH = module.chamfer_3DDist()
        print('Chamfer backend: SnowflakeNet CUDA extension', flush=True)
    return _CUDA_SEARCH


def _matched_distance(x, y, index):
    matched = y.gather(1, index.long().unsqueeze(-1).expand(-1, -1, 3))
    return torch.linalg.vector_norm(x.float() - matched.float(), dim=-1).mean()


def chamfer(a, b, one_way=False):
    if a.is_cuda and _BACKEND != 'chunked':
        try:
            search = _load_cuda_search()
        except Exception as exc:
            raise RuntimeError('CUDA Chamfer unavailable. Install/build the repository Chamfer extension, '
                               'or explicitly select --chamfer-backend chunked (slower).') from exc
        with torch.no_grad():
            _, _, index_a, index_b = search(a.detach().float().contiguous(),
                                           b.detach().float().contiguous())
        ab = _matched_distance(a, b, index_a)
        return ab if one_way else (ab + _matched_distance(b, a, index_b)) / 2
    if _BACKEND == 'cuda' and not a.is_cuda:
        raise ValueError('CUDA Chamfer requires CUDA tensors')
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
