import torch


def mse(x: torch.Tensor, target: torch.Tensor):
    assert x.shape[0] == target.shape[0]
    n = max(x.numel(), target.numel())
    return (x - target).pow(2).sum() / n


def snr(x: torch.Tensor, target: torch.Tensor):
    assert x.shape[0] == target.shape[0]
    scale = x.numel() // target.numel()
    noise = (x - target).pow(2).sum()
    signal = target.pow(2).sum() * scale
    SNR = 10 * (signal / noise).log10_()
    return SNR
