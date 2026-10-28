"""Compare CUDA nearest-neighbor loss with the exact chunked reference."""
import torch
from experiments.losses import chamfer, configure_chamfer, _matched_distance


def main():
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(11)
    for one_way in (False, True):
        a = torch.randn(2, 270, 3, device=device, requires_grad=True)
        b = torch.randn(2, 1050, 3, device=device, requires_grad=True)
        configure_chamfer('chunked')
        reference = chamfer(a, b, one_way)
        reference_grad = torch.autograd.grad(reference, (a, b))
        if device == 'cuda':
            configure_chamfer('cuda')
            result = chamfer(a, b, one_way)
        else:
            with torch.no_grad():
                distance = torch.cdist(a, b)
                ia, ib = distance.argmin(-1), distance.argmin(-2)
            ab = _matched_distance(a, b, ia)
            result = ab if one_way else (ab + _matched_distance(b, a, ib)) / 2
        gradients = torch.autograd.grad(result, (a, b))
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
        for actual, expected in zip(gradients, reference_grad):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    print('PASS: values and gradients; ' + ('CUDA extension checked' if device == 'cuda' else 'CPU gather path checked, CUDA unavailable'))


if __name__ == '__main__':
    main()
