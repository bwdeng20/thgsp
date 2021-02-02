def mse(x, target):
    n = max(x.numel(), target.numel())
    return (x - target).pow(2).sum() / n


def snr(x, target):
    noise = (x - target).pow(2).sum()
    signal = target.pow(2).sum()
    SNR = 10 * (signal / noise).log10_()
    return SNR
